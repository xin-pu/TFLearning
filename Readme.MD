# Opencv Call Model Generate by Tensorflow

```This is cmd reference```
>工具
* Tensorflow 2.5
* python 3.8
* OpenCVSharp


## Tensofflow






## 转换Tensorflow PB 模型 到 onnx

### 使用tf2onnx模块

```
    python -m tf2onnx.convert --saved-model D:\DataSets\Model --output resnet50.onnx
```

##### windows + Python3.8 直接转换很慢的处理办法
<https://github.com/onnx/tensorflow-onnx/issues/1557>
Protobuf has solved this with a cpp release for windows. Solve with:


```
     pip uninstall protobuf
     pip install protobuf
```



## OpenCV - DNN

```
var cnn = CvDnn.ReadNetFromOnnx(model);
if (cnn == null) return;
```

>注意点  
  
* 通过BlobFromImage/BlobFromImage转换图像到与tensorflowDims==4的张量形式  
* 如果Tensorflow训练的图像经过Resize 和 归一等预处理，在这里也需要对应。  
* swapRB，crop 训练时没有做其他处理，这里全部关掉 

```
var image = Cv2.ImRead(filename, ImreadModes.Grayscale);
var inputBlob = CvDnn.BlobFromImage(image, 1F / 255, new Size(64, 64), 
                                    swapRB: false, 
                                    crop: false);

net.SetInput(inputBlob);
var result = net.Forward();
PrintMatrixProb(result);
result.GetArray(out float[] data);
var max = data.ToList().IndexOf(data.Max()) + 1;
return max;
```
