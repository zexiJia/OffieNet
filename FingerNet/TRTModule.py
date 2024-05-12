import torch
import tensorrt as trt

def trt_version():
    return trt.__version__

def ReadEngineFile(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def default_input_names(num_inputs):
    return ["input_%d" % i for i in range(num_inputs)]

def default_output_names(num_outputs):
    return ["output_%d" % i for i in range(num_outputs)]

def torch_dtype_to_trt(dtype):
    if trt_version() >= '7.0' and dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)

def torch_device_to_trt(device):
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        if isinstance(self.engine, str): self.engine = ReadEngineFile(self.engine)
        if self.engine is not None:
            self.context = self.engine.create_execution_context()

        if isinstance(input_names, int): input_names = default_input_names(input_names)
        if isinstance(output_names, int): output_names = default_output_names(output_names)

        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        engine_bytes = state_dict[prefix + "engine"]

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].contiguous().data_ptr()

        self.context.execute_async(
            batch_size, bindings, torch.cuda.current_stream().cuda_stream
        )

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()

if __name__ == '__main__':

    from Dataset import NIST27
    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    from PostProcessing.MinutiaeRegressionTools import MinutiaeTools
    from PostProcessing.SegmentLabelTools import SegmentLabelTools
    from PostProcessing.EnhanceImageRegressionProducer import EnhanceImageProducer
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    producer = EnhanceImageProducer(dev = torch.device(0))
    label = SegmentLabelTools()

    d = NIST27('/Datasets/NISTSD27/matched/L')
    testd = DataLoader(d, 10, collate_fn=d.NIST27_collate_fn)

    def test_ORI(model):
        from PostProcessing.OrientationRegressionTools import OrientationTools

        T = OrientationTools(torch.device(0))
        res = T.Test_ori(model, testd)
        print(res)

    # model1 = TRTModule('./ori.trt', 1, 5)
    # model2 = TRTModule('./min.trt', 5, 4)
    model3 = TRTModule('./model.trt', 1, 1)
    # test_ORI(model)

    from Fingernet.Orientation_regression_wide_trt import Orientation
    from Fingernet.Minutiae_regression_wide import Minutiae

    import torch
    from utils import load_checkpoint

    # 0.58
    model1 = Orientation().cuda()
    load_checkpoint('./runs/best_wide.pth', model1)
    model2= Minutiae().cuda()
    load_checkpoint('/data/albert/FingerNet/runs/2021-03-30-03:52:40@pretrain/0_800.pth', model2)

    start = time.time()
    for data in tqdm(testd):

        imgs = data[0].cuda()
        ORI, Seg, tex, OF, QF = model1(imgs)
        Seg = torch.where(Seg>=0.5, 1., 0.)
        enh = producer.produce(imgs, ORI)
        c, w, h, o = model2(enh, tex, OF, ORI, Seg)
        Seg = label(Seg)
        embedding = model3(enh)
    torch.cuda.synchronize()
    end = time.time()
    print((end - start) / 258)
