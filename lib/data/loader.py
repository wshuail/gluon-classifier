from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class HybridTrainPipe(Pipeline):
    def __init__(self, rec_path, index_path, batch_size, input_size, num_gpus, num_threads, device_id):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[index_path],
                                     random_shuffle=True, shard_id=device_id, num_shards=num_gpus)
        # self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed",
        self.decode = ops.HostDecoderRandomCrop(device="cpu",
                                                 output_type=types.RGB,
                                                 random_aspect_ratio=[0.75, 1.25],
                                                 random_area = [0.08, 1.0],
                                                 num_attempts = 100)
        self.resize = ops.Resize(device="gpu", resize_x=input_size, resize_y=input_size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(input_size, input_size),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = images.gpu()
        images = self.resize(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels.gpu()]


class HybridValPipe(Pipeline):
    def __init__(self, rec_path, index_path, batch_size, input_size, num_gpus, num_threads, device_id):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[index_path],
                                     random_shuffle=False, shard_id=device_id, num_shards=num_gpus)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        # self.decode = ops.HostDecoder(device="cpu", output_type=types.RGB)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(input_size, input_size),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = images.gpu()
        output = self.cmnp(images)
        return [output, self.labels.gpu()]
