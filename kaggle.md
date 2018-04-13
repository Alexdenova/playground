

```python
import torch
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from IPython.display import clear_output
from ipywidgets import interact

from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row, gridplot
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()

torch.cuda.manual_seed_all(123)
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="b69c41b2-2e68-47a5-aaa5-e2c90cadd532">Loading BokehJS ...</span>
    </div>





```python
#Read csv
datasets = pd.read_csv('./diabetes.csv')
datasets.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Make train/valid/test set
x = datasets.drop('Outcome', axis=1).values
target = datasets['Outcome'].values
train_x, test_x, train_target, test_target = train_test_split(x, target, test_size=0.2, shuffle=True, random_state=123)
valid_x, test_x, valid_target, test_target = train_test_split(test_x, test_target, test_size=0.5, shuffle=True, random_state=123)
```


```python
#Utility function
def isnan(tensor):
    return tensor != tensor
def nv(arr):
    return Variable(torch.from_numpy(arr).float().cuda())
def nt(arr):
    return Variable(torch.from_numpy(arr).long().cuda())
def calc_lr(epoch_num):
    return 0.000000001 * 1.3 ** epoch_num
```


```python
#Numpy to Variable
train_x, valid_x, test_x = nv(train_x), nv(valid_x), nv(test_x)
train_target, valid_target, test_target = nt(train_target), nt(valid_target), nt(test_target)
```


```python
#Define your model
class model(nn.Module):
    def __init__(self, in_num):
        super(model, self).__init__()
        self.in_num = in_num
        self.L1 = nn.Linear(self.in_num, 512).cuda()
        self.L2 = nn.Linear(512, 128).cuda()
        self.L3 = nn.Linear(128, 2).cuda()
        self.B1 = nn.BatchNorm1d(512).cuda()
        self.B2 = nn.BatchNorm1d(128).cuda()
        
        nn.init.xavier_normal(self.L1.weight)
        nn.init.xavier_normal(self.L2.weight)
        nn.init.xavier_normal(self.L3.weight)
        

    def forward(self, inputs):

        h = F.relu(self.L1(inputs))
        h = self.B1(h)
        h = F.relu(self.L2(h))
        h = self.B2(h)
        h = self.L3(h)
        
        return h
```


```python
#Settings
batch_size = 32
num_input = train_x.size(1)

#Model

classifier = model(num_input)
torch.save(classifier.state_dict(), './init.pth') 
finder = model(num_input)
finder.load_state_dict(torch.load('./init.pth'))

#Loss function
criterion = nn.CrossEntropyLoss()
```


```python
#Verify finder's weight is the same as classifier's weight
classifier.L1.weight == finder.L1.weight
```




    Variable containing:
        1     1     1  ...      1     1     1
        1     1     1  ...      1     1     1
        1     1     1  ...      1     1     1
           ...          ⋱          ...       
        1     1     1  ...      1     1     1
        1     1     1  ...      1     1     1
        1     1     1  ...      1     1     1
    [torch.cuda.ByteTensor of size (512,8) (GPU 0)]




```python
##Optimal learning rates

#Log

find_logs = []

#Plotting

source_find = ColumnDataSource(data=dict(epoch=[], loss=[], lr=[]))
settings = dict(plot_width=480, plot_height=430, min_border=0)
pf = figure(title="Optimal learning rates", x_axis_label="epoch",y_axis_label="Loss", **settings)
pf.line(x='epoch', y='loss', source=source_find)

pf2 = figure(title="Corresponding Learning rate", x_axis_label="epoch", y_axis_label="Learning rate", **settings)
pf2.line(x='epoch', y='lr', source=source_find, color="orange")

tf = show(row(pf,pf2), notebook_handle=True)

#Config

train_v = train_x[:batch_size]
batch_num = len(train_v) // batch_size

optimizer = optim.Adam(finder.parameters(), lr=0.000000001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1.3, last_epoch=-1)

for epoch in tqdm(range(0,80)):
    
    #Training
    
    for batch in tqdm(range(0,batch_num), disable=True):
        
        start = batch*batch_size
        end = batch*batch_size + batch_size
        
        v = train_v[start:end]
        target = train_target[start:end]

        y_hat = finder(v)
        
        loss = criterion(y_hat, target)
        loss.backward()
        optimizer.step()
        finder.zero_grad()
    
    
    #Logiging 
    
    new_data_find = {'epoch' : [epoch], 'loss' : [loss.view(1).data[0]], 'lr' : [calc_lr(epoch)]}
    new_data_find_df = {'epoch' : epoch, 'loss' : loss.view(1).data[0], 'lr' : calc_lr(epoch)}
    source_find.stream(new_data_find)
    
    find_logs.append(new_data_find_df)
    df_find = pd.DataFrame(find_logs)
    df_find.to_csv("./find_logs.csv", index=False)
    
    scheduler.step()
    
    #Show plots
    
    push_notebook(handle=tf)

clear_output()

find_logs = pd.read_csv('./find_logs.csv')
source_find = ColumnDataSource(find_logs)

tf = show(row(pf,pf2))

print('Done')
```



<div class="bk-root">
    <div class="bk-plotdiv" id="6826002e-fa24-458a-a8e6-08bdd9ea0055"></div>
</div>




    Done



```python
#Log

train_logs = []
test_logs = []

#Plot

source_train = ColumnDataSource(data=dict(epoch=[], loss=[], precision=[], recall=[], f1=[], acc=[]))
source_test = ColumnDataSource(data=dict(epoch=[], loss=[], precision=[], recall=[], f1=[], acc=[]))

settings = dict(plot_width=480, plot_height=430, min_border=0)
p = figure(title="Cross Entropy Loss", x_axis_label="epoch",y_axis_label="Loss", **settings)
p.line(x='epoch', y='loss', source=source_train, legend="Train")
p.line(x='epoch', y='loss', source=source_test, legend="Valid", color="orange")

p2 = figure(title="Precision = tp/(tp+fp)", x_axis_label="epoch", y_axis_label="Precision", **settings)
p2.line(x='epoch', y='precision', source=source_train, legend="Train")
p2.line(x='epoch', y='precision', source=source_test, legend="Valid", color="orange")

p3 = figure(title="Recall = tp/(tp+fn)", x_axis_label="epoch", y_axis_label="Recall", **settings)
p3.line(x='epoch', y='recall', source=source_train, legend="Train")
p3.line(x='epoch', y='recall', source=source_test, legend="Valid", color="orange")

p4 = figure(title="F1 score = 2*(precision*recall)/(precision+recall)", x_axis_label="epoch", y_axis_label="F1 score", **settings)
p4.line(x='epoch', y='f1', source=source_train, legend="Train")
p4.line(x='epoch', y='f1', source=source_test, legend="Valid", color="orange")

p5 = figure(title="Accuracy", x_axis_label="epoch", y_axis_label="Accuracy", **settings)
p5.line(x='epoch', y='acc', source=source_train, legend="Train")
p5.line(x='epoch', y='acc', source=source_test, legend="Valid", color="orange")

grid = gridplot([[p, None], [p5, p4], [p2, p3]])

t = show(grid, notebook_handle=True)

#Config
batch_num = len(train_v) // batch_size


optimizer = optim.Adam(classifier.parameters(), lr=calc_lr(65))

for epoch in tqdm(range(0,100)):
    
    
    #Training
    
    for batch in tqdm(range(0,batch_num), disable=True):
        
        start = batch*batch_size
        end = batch*batch_size + batch_size
        
        v = train_x[start:end]
        target = train_target[start:end]

        y_hat = classifier(v)
        
        loss = criterion(y_hat, target)
        loss.backward()
        optimizer.step()
        classifier.zero_grad()
    
    
    #Train set Evaluation    
    
    precision_train = precision_score(target, torch.max(y_hat, 1)[1])
    recall_train = recall_score(target, torch.max(y_hat, 1)[1])
    f1_train = f1_score(target, torch.max(y_hat, 1)[1])
    acc_train = accuracy_score(target, torch.max(y_hat, 1)[1])
    
    new_data_train = {'epoch' : [epoch], 'loss' : [loss.view(1).data[0]], 'precision' : [precision_train], 'recall': [recall_train] ,'f1' :[f1_train], 'acc': [acc_train]}
    new_data_train_df = {'epoch' : epoch, 'loss' : loss.view(1).data[0], 'precision' : precision_train, 'recall': recall_train ,'f1' :f1_train, 'acc': acc_train}
    source_train.stream(new_data_train)
    
    train_logs.append(new_data_train_df)
    df_train = pd.DataFrame(train_logs)
    df_train.to_csv("./train_logs.csv", index=False)
    
    
    #Valid set Evaluation
 
    y_test_hat = classifier(test_x)
    loss_test = criterion(y_test_hat, test_target)
        
    precision_test = precision_score(test_target, torch.max(y_test_hat, 1)[1])
    recall_test = recall_score(test_target, torch.max(y_test_hat, 1)[1])
    f1_test = f1_score(test_target, torch.max(y_test_hat, 1)[1])
    acc_test = accuracy_score(test_target, torch.max(y_test_hat, 1)[1])

    new_data_test = {'epoch' : [epoch], 'loss' : [loss_test.view(1).data[0]], 'precision' : [precision_test], 'recall': [recall_test] ,'f1' :[f1_test], 'acc':[acc_test]}
    new_data_test_df = {'epoch' : epoch, 'loss' : loss_test.view(1).data[0], 'precision' : precision_test, 'recall': recall_test ,'f1' :f1_test, 'acc': acc_test}
    source_test.stream(new_data_test)
    
    test_logs.append(new_data_test_df)
    df_test = pd.DataFrame(test_logs)
    df_test.to_csv("./valid_logs.csv", index=False)
    
    
    #Show plots
    
    push_notebook(handle=t)
    
    
    #Save model per n epoch
    n = 100
    if (epoch % n) ==  0:
        torch.save(classifier.state_dict(), 'epoch{0}.pth' .format(epoch))

clear_output()

train_logs = pd.read_csv('./train_logs.csv')
test_logs = pd.read_csv('./valid_logs.csv')

source_train = ColumnDataSource(train_logs)
source_test = ColumnDataSource(test_logs)

t = show(grid, notebook_handle=True)

print('Done')
```



<div class="bk-root">
    <div class="bk-plotdiv" id="a8cac18b-e23e-4259-9560-716e17606c6f"></div>
</div>




    Done

