### Torchground

`$ nvidia-docker run -i -t -p 8888:8888 -v directory_pass:/notebooks relutropy/torchground /bin/bash`  
Change directry_pass to the directory pass you want to save files.  

In a container  

`root$ yum install -y git`  
`root$ cd notebooks/`  
`root$ git clone https://github.com/shllln/playground.git`  
`root$ jupyter notebook --allow-root`  

Go to http://localhost:8888/ and play.

#### Notebook lits  
- PimaIndiansDiabetes.ipynb (GPU only)  
