<h1 align="center">
  <img 
    width="64" 
    src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/dfb5c3b7b10e20878a3fee6e3b05660e4d3bd9d5/assets/Detective/Default/3D/detective_3d_default.png"/>
  <img 
    width="64" 
    src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/dfb5c3b7b10e20878a3fee6e3b05660e4d3bd9d5/assets/Artist/Default/3D/artist_3d_default.png"/>
    <p>MNIST GAN</p>
</h1>

![image](https://github.com/rgerd/feu/assets/4724014/8d4a6484-19d7-447d-8446-e6cb54ebecfe)
(Need to keep training but it looks a bit like this)

### Helpful links:
* [GanHacks](https://github.com/soumith/ganhacks)
* [Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic)

### Things I learned working on this:
* Sometimes eval() and train() behave completely differently?
    * Need to look into this more, but when I use .eval() on the discriminator when training the generator I get bad values.
    * There should be some kind of razor about the amount of time you spend debugging something... Like if you can't figure out the cause after a few days of looking and really applying yourself then more likely than not it's some tiny little thing.
* If your tanh is saturating it might be due to...
    * Weights initialized with too much magnitude.
    * Learning rate too high.
    * Loss gradients are pointing the wrong direction.
    * Gradients (and therefore weights, and therefore outputs) are exploding across the layers of your model.
* If the outputs of your CNN look repetitive, make sure your weights aren't super small, you could be crushing the feature vector--floating point is only so precise!
* Don't put strided convolutions in every layer, having a good amount of overlap allows you to diffuse information better (maybe).
* In deep networks it is likely the case that your first layers won't change much, since gradients will be small by then. Makes sense why resnets and batchnorm are useful.
* A lot of machine learning is infrastructural stuff around the model, and then tinkering with hyperparams. 
    * Try to modularize early and often.
    * Make your iteration loop as quick as possible.
