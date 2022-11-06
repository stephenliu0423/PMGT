This is the implementation of PMGT described in the paper: "[Pre-training Graph Transformer with Multimodal Side Information for Recommendation](https://doi.org/10.1145/3474085.3475709)" (In ACM MM2021).


## Environment
To run the code, you need the following dependencies
- pre-training:
    - Python 3
    - TensorFlow-gpu 1.13.1
    - graphlearn 1.0.1

- downstream:
    - Python 3
    - PyTorch 1.8.1

## Files in The Folder
- config_file/ : hyper-parameters;
- data/ : data pre-processing & pre-processed file;
- down_rec/ : implementation of downstream tasks;
- utils/ : optimization and bert modules and so on. 


## Quick Validation

The pre-trained representations of items in the video game dataset can be downloaded from [here](https://drive.google.com/file/d/1u9GmGizdsDe8FheYK6idflCpEy2WkVij/view?usp=share_link). Please move the unzipped files to the folder 'data/video/', and then run the codes of downstream tasks directly.

### Testing on Recommendation Task
Using the pre-trained item representations.  
` $ python run_rec.py --data_type video --pretrain 1 --lr 0.001 --l2_re 0`

Using the randomly initialized item representations.  
` $ python run_rec.py --data_type video --pretrain 0 --lr 0.001 --l2_re 0`

### Testing on CTR Prediction Task
Using the pre-trained item representations.  
` $ python run_ctr.py --data_type_video --pretrain 1 --lr 0.001 --l2_re 0.0001 `

Using the randomly initialized item representations.   
` $ python run_ctr.py --data_type_video --pretrain 0 --lr 0.001 --l2_re 0.0001 `

## Example of Running The Codes
### Data Preprocessing
The experimental datasets are collected from the [Amazon Review Datasets](https://nijianmo.github.io/amazon/index.html).
- Video Games
- Toys and Games
- Tools and Home Improvement

Using the original data to build the pre-training graph dataset and downstream task dataset.  
` $ python data_process.py `

Note that the experimental datasets used in the original paper are processed based on some internal APIs. Thus, there exist some difference between the following experimental statistics and the statistics reported in the original paper.  


### Statistics of Experimental Datasets

<table>
  <tr>
    <td rowspan="2" style="text-align:center">Datasets</td>
    <td colspan="3" style="text-align:center">Data for Downstream tasks</td>
    <td colspan="2" style="text-align:center">Item Graph</td>
    <td rowspan="2" style="text-align:center">Threshold</td>
  </tr>
  <tr>
    <td style="text-align:center"># Users</td>
    <td style="text-align:center" ># Items</td>
    <td style="text-align:center"># Interact.</td>
    <td style="text-align:center"># Nodes</td>
    <td style="text-align:center"># Edges</td>
  </tr>
  <tr>
    <td style="text-align:center">VG</td>
    <td style="text-align:center">27,988</td>
    <td style="text-align:center">6,551</td>
    <td style="text-align:center">98,278</td>
    <td style="text-align:center">7,252</td>
    <td style="text-align:center">88,606</td>
    <td style="text-align:center">3</td>
  </tr>
  <tr>
    <td style="text-align:center">TG</td>
    <td style="text-align:center">118,153</td>
    <td style="text-align:center">6,238</td>
    <td style="text-align:center">294,507</td>
    <td style="text-align:center">6,451</td>
    <td style="text-align:center">15,363</td>
    <td style="text-align:center">4</td>
  </tr>
  <tr>
    <td style="text-align:center">THI</td>
    <td style="text-align:center">164,717</td>
    <td style="text-align:center">5,751</td>
    <td style="text-align:center">431,455</td>
    <td style="text-align:center">5,982</td>
    <td style="text-align:center">12,927</td>
    <td style="text-align:center">3</td>
  </tr>
</table>


### Pre-training
#### Pre-training PMGT  
` $ python main.py --data_type video --is_train 1`

#### Saving Item Representations Pre-trained by PMGT  
` $ python main.py --data_type video --is_train 0`

### Testing on Downstream Tasks  
See the detailed in Quick Validation

### Experiment Results 

<table>
  <tr>
    <td rowspan="2" style="text-align:center">Datasets</td>
    <td rowspan="2" style="text-align:center">Methods</td>
    <td colspan="4" style="text-align:center">Top-N Recommendation</td>
  </tr>
  <tr>
    <td style="text-align:center">REC-R@10</td>
    <td style="text-align:center">REC-R@20</td>
    <td style="text-align:center">REC-N@10</td>
    <td style="text-align:center">REC-N@20</td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align:center">VG</td>
  </tr>
  <tr>
    <td style="text-align:center">NCF</td>
    <td style="text-align:center">0.1698</td>
    <td style="text-align:center">0.2510</td>
    <td style="text-align:center">0.0970</td>
    <td style="text-align:center">0.1192</td>
  </tr>
  <tr>
    <td style="text-align:center">NCF-PMGT</td>
    <td style="text-align:center">0.2588</td>
    <td style="text-align:center">0.3518</td>
    <td style="text-align:center">0.1688</td>
    <td style="text-align:center">0.1945</td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align:center">TG</td>
  </tr>
  <tr>
    <td style="text-align:center">NCF</td>
    <td style="text-align:center">0.2598</td>
    <td style="text-align:center">0.3295</td>
    <td style="text-align:center">0.1942</td>
    <td style="text-align:center">0.2129</td>
  </tr>
  <tr>
    <td style="text-align:center">NCF-PMGT</td>
    <td style="text-align:center">0.2926</td>
    <td style="text-align:center">0.3682</td>
    <td style="text-align:center">0.2194</td>
    <td style="text-align:center">0.2397</td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align:center">THI</td>
  </tr>
  <tr>
    <td style="text-align:center">NCF</td>
    <td style="text-align:center">0.2687</td>
    <td style="text-align:center">0.3188</td>
    <td style="text-align:center">0.2232</td>
    <td style="text-align:center">0.2367</td>
  </tr>
  <tr>
    <td style="text-align:center">NCF-PMGT</td>
    <td style="text-align:center">0.2909</td>
    <td style="text-align:center">0.3509</td>
    <td style="text-align:center">0.2390</td>
    <td style="text-align:center">0.2552</td>
  </tr>
</table>
