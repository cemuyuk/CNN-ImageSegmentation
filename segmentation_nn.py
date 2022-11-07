"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################
        from torchvision import models
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
           
        self.features = models.alexnet(pretrained = True, progress = True).features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 4096, 6),
            nn.ReLU(inplace=True),
            
            
            nn.Dropout(p=0.5),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(4096, num_classes, 1),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=5, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=3, mode='bilinear')
        )
        
        #self.model = lraspp_mobilenet_v3_large(num_classes=num_classes)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        #x = self.model(x)['out']

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x
    
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            SegmentationNN.parameters(self),
            lr=self.hparams["learning_rate"],
            momentum=self.hparams["momentum"]
        )
        return optim

    def general_step(self, batch, batch_idx, mode):

        loss_fn = F.torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

        images, targets = batch
        outputs = self.forward(images)

        loss = loss_fn(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        self.log("loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return {'val_loss': loss}

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
