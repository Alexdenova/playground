## Playground for ML
Here is the playground for python, machine learning, deep learning, and what you are interested in.  

### Docker envriroments

> ##### Pure enviroments

For CPU  
`$ docker run -i -t relutropy/playground /bin/bash`  

For GPU   

nvidia-docker version 2.0   
  `$ docker run --runtime=nvidia -i -t relutropy/playground /bin/bash`    
nvidia-docker   
  `$ nvidia-docker run -i -t relutropy/playground /bin/bash`     

> ##### Pytorch enviroments

For CPU  
`$ docker run -i -t relutropy/torchground /bin/bash`  

For GPU   

nvidia-docker version 2.0   
  `$ docker run --runtime=nvidia -i -t relutropy/torchground /bin/bash`    
nvidia-docker   
  `$ nvidia-docker run -i -t relutropy/torchground /bin/bash`     
