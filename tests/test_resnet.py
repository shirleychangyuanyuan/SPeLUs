from case_script import TestPythonScript
import os
import shutil


class ResnetTest(TestPythonScript):

    @property
    def script(self):
        #return '../examples/ResNet/imagenet-resnet.py'
        return '../examples/ResNet/cifar10_resnext_29_new.py'

    # def test(self):
    #     self.assertSurvive(self.script, args=['--data .',
    #                                           '--gpu 0', '--fake', '--data_format NCHW'], timeout=10)

    def test(self):
        self.assertSurvive(self.script, args=['--load ','--gpu 0'], timeout=10)

    def tearDown(self):
        super(ResnetTest, self).tearDown()
        if os.path.isdir('ilsvrc'):
            shutil.rmtree('ilsvrc')
