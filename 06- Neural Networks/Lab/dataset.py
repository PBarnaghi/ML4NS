import torch
import numpy as np
import os
import typing
import typing
import ast
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.datasets.utils import download_and_extract_archive
import joblib
import tqdm
try:
    import wfdb
    wfdb_import_error = False
except ImportError:
    wfdb_import_error = True


class PTB_XL(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path:str='./',
        train:bool=True,
        sampling_rate:typing.Literal[100, 500]=100,
        binary:bool=False,
        subset=False,
        ):
        '''
        ECG Data, as described here: https://physionet.org/content/ptb-xl/1.0.2/.
        A positive class when :code:`binary=True`, indicates that
        the ECG Data is abnormal.
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = PTB_XL(
            ...     data_path='../../data/', 
            ...     train=True, 
            ...     sampling_rate=500,
            ...     )

        
        
        Arguments
        ---------
        
        - data_path: str, optional:
            The path that the data is saved
            or will be saved. 
            Defaults to :code:`'./'`.
        
        - train: bool, optional:
            Whether to load the training or testing set. 
            Defaults to :code:`True`.
        
        - sampling_rate: typing.Literal[100, 500], optional:
            The sampling rate. This should be
            in :code:`[100, 500]`. 
            Defaults to :code:`100`.
        
        - binary: bool, optional:
            Whether to return classes based on whether the 
            ecg is normal or not, and so a binary classification
            problem.
            Defaults to :code:`False`.
        
        - subset: bool, optional:
            If :code:`True`, only the first 1000 items
            of the training and test set will be returned.
            Defaults to :code:`False`.
        
        
        '''

        if wfdb_import_error:
            raise ImportError(
                'Please install wfdb before using this dataset. Use pip install wfdb.'
                )

        assert sampling_rate in [100, 500], \
            "Please choose sampling_rate from [100, 500]"
        assert type(train) == bool, "Please use train = True or False"

        self.data_path = data_path
        self.download()
        self.data_path = os.path.join(
            self.data_path, 
            'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2/',
            )

        self.train=train
        self.sampling_rate = sampling_rate
        self.binary = binary
        self.meta_data = pd.read_csv(self.data_path+'ptbxl_database.csv')
        self.meta_data['scp_codes'] = (self.meta_data
            ['scp_codes']
            .apply(lambda x: ast.literal_eval(x))
            )
        self.aggregate_diagnostic() # create diagnostic columns
        self.feature_names = [
            'I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
            ]

        if self.train:
            self.meta_data = self.meta_data.query("strat_fold != 10")
            if subset:
                self.meta_data = self.meta_data.iloc[:1000]
        else:
            self.meta_data = self.meta_data.query("strat_fold == 10")
            if subset:
                self.meta_data = self.meta_data.iloc[:1000]
        
        if binary:
            self.targets = self.meta_data[['NORM', 'CD', 'HYP', 'MI', 'STTC']].values
            self.targets = 1-self.targets[:,0]
        else:
            self.targets = self.meta_data[['NORM', 'CD', 'HYP', 'MI', 'STTC']].values


        return

    def _check_exists(self):
        folder = os.path.join(
            self.data_path, 
            'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2',
            )
        return os.path.exists(folder)
        
    def download(self):
        
        if self._check_exists():
            print("Files already downloaded.")
            return

        download_and_extract_archive(
            url='https://physionet.org/static'\
                '/published-projects/ptb-xl/'\
                'ptb-xl-a-large-publicly-available'\
                '-electrocardiography-dataset-1.0.2.zip',
            download_root=self.data_path,
            extract_root=self.data_path,
            filename='ptbxl.zip',
            remove_finished=True
            )

        return

    @staticmethod
    def single_diagnostic(y_dict, agg_df):
        tmp = []
        for key in y_dict.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def aggregate_diagnostic(self):
        agg_df = pd.read_csv(self.data_path +'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        self.meta_data['diagnostic_superclass'] = (self.meta_data
            ['scp_codes']
            .apply(
                self.single_diagnostic, 
                agg_df=agg_df,
                )
            )
        mlb = MultiLabelBinarizer()
        self.meta_data = self.meta_data.join(
            pd.DataFrame(
                mlb.fit_transform(
                    self.meta_data.pop('diagnostic_superclass')
                    ),
                columns=mlb.classes_,
                index=self.meta_data.index,
                )
            )
        return

    def __getitem__(self, index):

        data = self.meta_data.iloc[index]

        if self.sampling_rate == 100:
            f = data['filename_lr']
            x = wfdb.rdsamp(self.data_path+f)
        elif self.sampling_rate == 500:
            f = data['filename_hr']
            x = wfdb.rdsamp(self.data_path+f)
        x = torch.tensor(x[0]).transpose(0,1).float()
        y = torch.tensor(
            data
            [['NORM', 'CD', 'HYP', 'MI', 'STTC']]
            .values
            .astype(np.int64)
            )
        if self.binary:
            y = 1-y[0]

        return x, y
    
    def __len__(self):
        return len(self.meta_data)
    














class MemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset:torch.utils.data.Dataset,
        now:bool=True,
        verbose:bool=True,
        n_jobs:int=1,
        ):
        '''
        This dataset allows the user
        to wrap another dataset and 
        load all of the outputs into memory,
        so that they are accessed from RAM 
        instead of storage. All attributes of
        the original dataset will still be available, except
        for :code:`._dataset` and :code:`._data_dict` if they 
        were defined.
        It also allows the data to be saved in memory right
        away or after the data is accessed for the first time.
               
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = MemoryDataset(dataset, now=True)
        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset: 
            The dataset to wrap and add to memory.
        
        - now: bool, optional:
            Whether to save the data to memory
            right away, or the first time the 
            data is accessed. If :code:`True`, then
            this initialisation might take some time
            as it will need to load all of the data.
            Defaults to :code:`True`.
        
        - verbose: bool, optional:
            Whether to print progress
            as the data is being loaded into
            memory. This is ignored if :code:`now=False`.
            Defaults to :code:`True`.
        
        - n_jobs: int, optional:
            The number of parallel operations when loading 
            the data to memory.
            Defaults to :code:`1`.
        
        
        '''

        self._dataset = dataset
        self._data_dict = {}
        if now:

            pbar = tqdm.tqdm(
                total = len(dataset),
                desc='Loading into memory',
                disable=not verbose,
                smoothing=0,
                )

            def add_to_dict(index):
                for ni, i in enumerate(index):
                    self._data_dict[i] = dataset[i]
                    pbar.update(1)
                    pbar.refresh()
                return None

            all_index = np.arange(len(dataset))
            index_list = [all_index[i::n_jobs] for i in range(n_jobs)]

            joblib.Parallel(
                n_jobs=n_jobs,
                backend='threading',
                )(
                    joblib.delayed(add_to_dict)(index)
                    for index in index_list
                    )
            
            pbar.close()

        return

    def __getitem__(self, index):

        if index in self._data_dict:
            return self._data_dict[index]
        else:
            output = self._dataset[index]
            self._data_dict[index] = output
            return output
    
    def __len__(self):
        return len(self._dataset)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError


