# 使用Keras的预训练模型
import tensorflow as tf
from tensorflow import keras

# 加载在ImageNet上预训练的ResNet-50模型
model = keras.applications.resnet50.ResNet50(weights="imagenet")

# 保证图片有正确的大小。ResNet-50模型要用224 × 224像素的图片
# images_resized = tf.image.resize(images, [224, 224])
# 提示：tf.image.resize()不会保留宽高比。如果需要，可以裁剪图片为合适的宽高比之后，再进行缩放。
# 两步可以通过tf.image.crop_and_resize()来实现。
images_resized = tf.image.crop_and_resize(images, [224, 224])

# 预训练模型的图片要经过特别的预处理。在某些情况下，要求输入是0到1，有时是-1到1，等等。
# 每个模型提供了一个preprocess_input()函数，来对图片做预处理。
# 这些函数假定像素值的范围是0到255，因此需要乘以255（因为之前将图片缩减到0和1之间）
inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)

# 和通常一样，输出Y_proba是一个矩阵，每行是一张图片，每列是一个类（这个例子中有1000类）。
Y_proba = model.predict(inputs)

# 展示top K 预测
# 正确的类（monastery 和 daisy）出现在top3的结果中。考虑到，这是从1000个类中挑出来的，结果相当不错。
top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print("Image #{}".format(image_index))
    for class_id, name, y_proba in top_K[image_index]:
        print("  {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
    print()

# 可以看到，使用预训练模型，可以非常容易的创建出一个效果相当不错的图片分类器。
# keras.applications中其它视觉模型还有几种ResNet的变体，GoogLeNet的变体（比如Inception-v3 和 Xception），
# VGGNet的变体，MobileNet和MobileNetV2（移动设备使用的轻量模型）。
