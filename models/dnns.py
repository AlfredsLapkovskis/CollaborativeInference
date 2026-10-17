import os
from abc import ABC, abstractmethod


class DNN(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def load_from_file(self, path: str):
        pass

    # To be used only on the server side!
    @abstractmethod
    def load_from_hub(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def predict_proba(self, input_data):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    @abstractmethod
    def predict_proba_batch(self, input_data_batch):
        pass

    @abstractmethod
    def predict_batch(self, input_data_batch):
        pass


class PyTorchVisionDNN(DNN):

    def __init__(self, device=None):
        self.model = None
        self.device = device

    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    def load_from_file(self, path: str):
        import torch
        self.model = torch.load(path, weights_only=False)
        self._to_device(self.model)

    def load_from_hub(self):
        self.model = self.build_model(weights=self.get_weights())
        self._to_device(self.model)

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        import torch
        torch.save(self.model, path)

    def predict_proba(self, input_data):
        return self._predict_proba_batch([input_data])[0]
        # return self._predict_proba(input_data).detach().cpu().numpy()

    def predict(self, input_data):
        return self.predict_batch([input_data])[0]
        # prediction_proba = self._predict_proba(input_data)
        # return prediction_proba.argmax().item()

    def predict_proba_batch(self, input_data_batch):
        return self._predict_proba_batch(input_data_batch).detach().cpu().numpy()

    def predict_batch(self, input_data_batch):
        proba = self._predict_proba_batch(input_data_batch)
        return proba.argmax(dim=1).detach().cpu().numpy().tolist()

    def _predict_proba(self, input_data):
        return self._predict_proba_batch([input_data])[0]
        # transforms = self.get_weights().transforms()
        # batch = transforms(input_data).unsqueeze(0)
        # batch = self._to_device(batch)
        # import torch
        # with torch.no_grad():
        #     prediction_proba = self.model(batch).squeeze(0).softmax(0)
        # return prediction_proba

    def _predict_proba_batch(self, input_data_batch):
        import torch
        transforms = self.get_weights().transforms()
        tensors = [transforms(img) for img in input_data_batch]
        batch = torch.stack(tensors, dim=0)
        batch = self._to_device(batch)
        with torch.inference_mode():
            predictions_proba = self.model(batch).softmax(dim=1)
        return predictions_proba

    def _to_device(self, object):
        if self.device is not None:
            import torch
            return object.to(torch.device(self.device))
        return object


class GoogleNetModel(PyTorchVisionDNN):
    def get_name(self):
        return "googlenet"

    def build_model(self, *args, **kwargs):
        from torchvision.models import googlenet
        return googlenet(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import GoogLeNet_Weights
        return GoogLeNet_Weights.IMAGENET1K_V1


class DenseNet201Model(PyTorchVisionDNN):
    def get_name(self):
        return "densenet201"

    def build_model(self, *args, **kwargs):
        from torchvision.models import densenet201
        return densenet201(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import DenseNet201_Weights
        return DenseNet201_Weights.IMAGENET1K_V1


class DenseNet169Model(PyTorchVisionDNN):
    def get_name(self):
        return "densenet169"

    def build_model(self, *args, **kwargs):
        from torchvision.models import densenet169
        return densenet169(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import DenseNet169_Weights
        return DenseNet169_Weights.IMAGENET1K_V1


class ResNet152Model(PyTorchVisionDNN):
    def get_name(self):
        return "resnet152"

    def build_model(self, *args, **kwargs):
        from torchvision.models import resnet152
        return resnet152(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import ResNet152_Weights
        return ResNet152_Weights.IMAGENET1K_V2


class VitB16Model(PyTorchVisionDNN):
    def get_name(self):
        return "vit-b-16"

    def build_model(self, *args, **kwargs):
        from torchvision.models import vit_b_16
        return vit_b_16(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import ViT_B_16_Weights
        return ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1


class VitB32Model(PyTorchVisionDNN):
    def get_name(self):
        return "vit-b-32"

    def build_model(self, *args, **kwargs):
        from torchvision.models import vit_b_32
        return vit_b_32(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import ViT_B_32_Weights
        return ViT_B_32_Weights.IMAGENET1K_V1


class VGG16Model(PyTorchVisionDNN):
    def get_name(self):
        return "vgg16"

    def build_model(self, *args, **kwargs):
        from torchvision.models import vgg16
        return vgg16(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import VGG16_Weights
        return VGG16_Weights.IMAGENET1K_V1


class VGG19Model(PyTorchVisionDNN):
    def get_name(self):
        return "vgg19"

    def build_model(self, *args, **kwargs):
        from torchvision.models import vgg19
        return vgg19(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import VGG19_Weights
        return VGG19_Weights.IMAGENET1K_V1


class VGG13Model(PyTorchVisionDNN):
    def get_name(self):
        return "vgg13"

    def build_model(self, *args, **kwargs):
        from torchvision.models import vgg13
        return vgg13(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import VGG13_Weights
        return VGG13_Weights.IMAGENET1K_V1


class VitL16Model(PyTorchVisionDNN):
    def get_name(self):
        return "vit-l-16"

    def build_model(self, *args, **kwargs):
        from torchvision.models import vit_l_16
        return vit_l_16(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import ViT_L_16_Weights
        return ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1


class VitL32Model(PyTorchVisionDNN):
    def get_name(self):
        return "vit-l-32"

    def build_model(self, *args, **kwargs):
        from torchvision.models import vit_l_32
        return vit_l_32(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import ViT_L_32_Weights
        return ViT_L_32_Weights.IMAGENET1K_V1


class VitH14Model(PyTorchVisionDNN):
    def get_name(self):
        return "vit-h-14"

    def build_model(self, *args, **kwargs):
        from torchvision.models import vit_h_14
        return vit_h_14(*args, **kwargs)

    def get_weights(self):
        from torchvision.models import ViT_H_14_Weights
        return ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1


def get_all_pytorch_models(device=None) -> list[PyTorchVisionDNN]:
    return [
        # Small (approx. 20-80Mb)
        GoogleNetModel(device=device),
        DenseNet201Model(device=device),
        DenseNet169Model(device=device),

        # Bigger (approx. 200-300Mb)
        ResNet152Model(device=device),
        VitB16Model(device=device),
        VitB32Model(device=device),

        # Big (approx. 500Mb)
        VGG16Model(device=device),
        VGG19Model(device=device),
        VGG13Model(device=device),

        # Large (approx. 1-2Gb+)
        VitL16Model(device=device),
        VitL32Model(device=device),
        VitH14Model(device=device),
    ]
