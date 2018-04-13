## Playground for ML
Here is the playground for python, machine learning, deep learning, and what you are interested in.  

### Docker envriroments

> ##### Pure enviroment

For CPU  
`$ docker run -i -t relutropy/playground /bin/bash`  

For GPU   
nvidia-docker version 2.0   
`$ docker run --runtime=nvidia -i -t relutropy/playground /bin/bash`  
nvidia-docker  
`$ nvidia-docker run -i -t relutropy/playground /bin/bash`  

  

> ##### Pytorch enviroment

For CPU  
`$ docker run -i -t -p 8888:8888 -v directory_pass:/notebooks relutropy/torchground /bin/bash`  
`root$ jupyter notebook --allow-root`  
Browse http://localhost:8888/ and enter in notebooks directory.

For GPU   
nvidia-docker version 2.0   
`$ docker run --runtime=nvidia -i -t -p 8888:8888 -v directory_pass:/notebooks relutropy/torchground /bin/bash`  
`root$ jupyter notebook --allow-root`  

nvidia-docker   
`$ nvidia-docker run -i -t -p 8888:8888 -v directory_pass:/notebooks relutropy/torchground /bin/bash`     
`root$ jupyter notebook --allow-root`  


