
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
from sgmsvs.MSS_dataset import MSSMUSDBDataset, MSSMoisesDBDataset, MSSMUSMoisDBDataset
from sgmsvs.sgmse.util.other import pad_spec
import numpy as np
import torch.nn.functional as F

#TODO: add support for validation with musmoisdb-dataset => does not work for some reason! without flag use_musdb_test_as_valid 

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', normalize="noisy", spec_transform=None,
            stft_kwargs=None, **ignored_kwargs):

        # Read file paths according to file naming format.
        if format == "default" or format=="noise":
            self.clean_files = []
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "*.wav")))
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "**", "*.wav")))
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "*.wav")))
            self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "**", "*.wav")))
        elif format == "reverb":
            self.clean_files = []
            self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "*.wav")))
            self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "**", "*.wav")))
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "*.wav")))
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "**", "*.wav")))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"
        
    def __getitem__(self, i):
        x, _ = load(self.clean_files[i])
        y, _ = load(self.noisy_files[i])

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)


        return X, Y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)

class MSSSpecs(Dataset):


    def __init__(self, 
                 data_dir, 
                 samples_per_track, 
                 subset, 
                 dataset_str,
                 valid_split,
                 rand_seed, 
                 target_str, 
                 random_mix_flag, 
                 augmentation_flag,
                 enforce_full_mix_percentage, 
                 duration, 
                 dummy, 
                 normalize,
                 spec_transform=None,
                 stft_kwargs=None,
                 num_frames=626,
                 train_mono=False,
                 pad_mix=True,
                 **ignored_kwargs):

        # Read file paths according to file naming format.
        if dataset_str == "musdb":
            train_name_list = []
            test_name_list = []
            train_name_list += sorted(glob(join(data_dir, "musdb18hq", "train", "**")))#sorted(glob(join(data_dir, "musdb18hq", "train", "**", "*.wav")))
            n_valid_files = int(valid_split * len(train_name_list))
            np.random.seed(rand_seed)
            idx_array = np.arange(len(train_name_list))
            valid_idx = np.random.choice(idx_array, n_valid_files, replace=False)
            valid_name_list = np.array(train_name_list)[valid_idx].tolist()#[train_name_list.pop(i) for i in valid_idx]
            train_name_list = np.array(train_name_list)
            train_name_list = np.delete(train_name_list, valid_idx).tolist()
            np.random.seed()
            test_name_list += sorted(glob(join(data_dir, "musdb18hq", "test", "**")))

            if subset == "train":
                file_list = train_name_list
                valid_flag = False
            elif subset == "valid":
                file_list = valid_name_list
                valid_flag = True
            elif subset == "test":
                file_list = test_name_list
                valid_flag = True

            self.dataset = MSSMUSDBDataset(file_list, target_str, 
                                           random_mix_flag, augmentation_flag, 
                                           enforce_full_mix_percentage,
                                           duration, samples_per_track, 
                                           valid_flag, rand_seed)

#            print("xop")

        elif dataset_str == "moisesdb":
            train_name_list = []
            test_name_list = []
            train_name_list += sorted(glob(join(data_dir, "moisesdb", "**")))#sorted(glob(join(data_dir, "musdb18hq", "train", "**", "*.wav")))
            n_valid_files = int(valid_split * len(train_name_list))
            np.random.seed(rand_seed)
            idx_array = np.arange(len(train_name_list))
            valid_idx = np.random.choice(idx_array, n_valid_files, replace=False)
            valid_name_list = np.array(train_name_list)[valid_idx].tolist()#[train_name_list.pop(i) for i in valid_idx]
            train_name_list = np.array(train_name_list)
            train_name_list = np.delete(train_name_list, valid_idx).tolist()
            np.random.seed()
            test_name_list += sorted(glob(join(data_dir, "musdb18hq", "test", "**")))

            if subset == "train":
                file_list = train_name_list
                valid_flag = False
            elif subset == "valid":
                file_list = valid_name_list
                valid_flag = True
            elif subset == "test":
                file_list = test_name_list
                valid_flag = True

            self.dataset = MSSMoisesDBDataset(file_list, target_str, 
                                              random_mix_flag, augmentation_flag, 
                                              duration, samples_per_track, 
                                              valid_flag, enforce_full_mix_percentage,
                                              rand_seed)

#            print("stop")

        elif dataset_str == "musmoisdb":
            moises_train_name_list = []
            moises_train_name_list += sorted(glob(join(data_dir, "moisesdb", "**")))#sorted(glob(join(data_dir, "musdb18hq", "train", "**", "*.wav")))
            musdb_train_name_list = []
            musdb_train_name_list += sorted(glob(join(data_dir, "musdb18hq", "train", "**")))#sorted(glob(join(data_dir, "musdb18hq", "train", "**", "*.wav")))
            n_valid_files = int(valid_split * (len(musdb_train_name_list)+len(moises_train_name_list)))
            mus_mois_ratio = len(musdb_train_name_list)/len(moises_train_name_list)
            n_mus_valid_files = int(n_valid_files*mus_mois_ratio)
            n_mois_valid_files = n_valid_files-n_mus_valid_files
            np.random.seed(rand_seed)
            mois_idx_array = np.arange(len(moises_train_name_list))
            mois_valid_idx = np.random.choice(mois_idx_array, n_mois_valid_files, replace=False)
            mois_valid_name_list = np.array(moises_train_name_list)[mois_valid_idx].tolist()#[train_name_list.pop(i) for i in valid_idx]
            moises_train_name_list = np.array(moises_train_name_list)
            moises_train_name_list = np.delete(moises_train_name_list, mois_valid_idx).tolist()

            musdb_idx_array = np.arange(len(musdb_train_name_list))
            musdb_valid_idx = np.random.choice(musdb_idx_array, n_mus_valid_files, replace=False)
            musdb_valid_name_list = np.array(musdb_train_name_list)[musdb_valid_idx].tolist()#[train_name_list.pop(i) for i in valid_idx]
            musdb_train_name_list = np.array(musdb_train_name_list)
            musdb_train_name_list = np.delete(musdb_train_name_list, musdb_valid_idx).tolist()
            np.random.seed()
            test_name_list = []
            test_name_list += sorted(glob(join(data_dir, "musdb18hq", "test", "**")))

            if subset == "train":
                moises_file_list = moises_train_name_list
                musdb_file_list = musdb_train_name_list
                valid_flag = False
            elif subset == "valid":
                moises_file_list = mois_valid_name_list
                musdb_file_list = musdb_valid_name_list
                valid_flag = True
            elif subset == "test":
                file_list = test_name_list
                valid_flag = True

            if subset != "test":
                musdb_samples_per_track = samples_per_track 
                moises_samples_per_track = int(np.ceil(mus_mois_ratio*samples_per_track))
                self.dataset = MSSMUSMoisDBDataset(moises_file_list,
                                                   moises_samples_per_track,
                                                   enforce_full_mix_percentage,
                                                   musdb_file_list,
                                                   musdb_samples_per_track,
                                                   valid_flag,
                                                   target_str, 
                                                   random_mix_flag, 
                                                   augmentation_flag, 
                                                   duration,
                                                   rand_seed)
            else:
                self.dataset = MSSMUSDBDataset(file_list, target_str, 
                                random_mix_flag, augmentation_flag, 0,
                                duration, samples_per_track, 
                                valid_flag, rand_seed)

        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")
        
        self.subset = subset
        self.dummy = dummy
        self.num_frames = num_frames
#        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.train_mono = train_mono
        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"
        self.pad_mix = pad_mix 
        self.dataset_str = dataset_str

    def __getitem__(self, i):

        y, x, target_rms = self.dataset[i]


        # formula applies for center=True
        target_len = (self.num_frames ) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad != 0:

            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            if self.pad_mix:
                y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')


        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        if x.isnan().any():
            print("x contain NaNs")
            breakpoint()
        
        if y.isnan().any():
            print("y contain NaNs")
            breakpoint()

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        
        if self.spec_transform is not None:
            X, Y = self.spec_transform(X), self.spec_transform(Y)
#        else:
            # just return audio for validation and test set
#            X = x
#            Y = y
        
        if X.isnan().any():
            print("X contain NaNs")
            breakpoint()
        
        if Y.isnan().any():
            print("Y contain NaNs")
            breakpoint()

        if self.train_mono:
            ch_id = np.random.randint(0, 2)
            return X[ch_id].unsqueeze(0), Y[ch_id].unsqueeze(0), x[ch_id].unsqueeze(0), y[ch_id].unsqueeze(0)
        else:
            return X, Y, x, y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.dataset)/200)
        else:
            return int(len(self.dataset))

class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, choices=("MSS", "default", "reverb"), default="MSS", help="Read file paths according to file naming format.")
        parser.add_argument("--dataset_str", type=str, choices=("musdb", "moisesdb", "musmoisdb"), default="musdb", help="Dataset to use. Default is musdb.")
        parser.add_argument("--target_str", type=str, choices=("vocals", "bass", "drums", "other", "all"), default="vocals", help="Target source to extract. 'vocals' by default.")
        parser.add_argument("--samples_per_track", type=int, default=1, help="How often each track is used in the dataset. Allows for oversampling of the dataset. 1 by default")
        parser.add_argument("--valid_split", type=float, default=0.1, help="Fraction of the training set to use for validation. 0.1 by default.")
        parser.add_argument("--use_musdb_test_as_valid", action="store_true", default=False, help="Use musdb test set as validation")
        parser.add_argument("--random_mix", action="store_true", default=False, help="Set for random mixing of sources.")
        parser.add_argument("--add_augmentation", action="store_true", default=False, help="Set for data augmentation.")
        parser.add_argument("--full_mix_percentage", type=float, default=0.7, help="Percentage of data with non-silent sources. 0.7 by default.")
        parser.add_argument("--rand_seed", default=13, type=int, help="Random seed that defines train/valid split.")
        parser.add_argument("--duration", type=float, default=5.0, help="Duration of the audio files. 5 seconds by default.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--train_mono", action="store_true", help="Use only the first channel of the audio files.")
        parser.add_argument("--n_fft", type=int, default=1534, help="Number of FFT bins. 1534 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=384, help="Window hop length. 384 by default.")
#        parser.add_argument("--num_frames", type=int, default=626, help="Number of frames for the dataset. 626 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for DataLoaders. 8 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.065, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.667, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        return parser

    def __init__(
        self, base_dir, samples_per_track=1, valid_split=0.1, format='MSS', dataset_str='musdb', target_str='vocals', random_mix=False, 
        add_augmentation=False, full_mix_percentage=0.7, use_musdb_test_as_valid=False, duration=5.0, rand_seed=13, batch_size=8, train_mono=False,
        n_fft=1534, hop_length=384, sr=48000, window='hann',#, num_frames=626
        num_workers=8, dummy=False, spec_factor=0.065, spec_abs_exponent=0.667,
        gpu=True, normalize='noisy', transform_type="exponent", **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.samples_per_track = samples_per_track
        self.valid_split = valid_split
        self.rand_seed = rand_seed
        self.format = format
        self.dataset_str = dataset_str
        self.target_str = target_str
        self.random_mix = random_mix
        self.aug_flag = add_augmentation
        self.full_mix_percentage = full_mix_percentage
        self.use_musdb_test_as_valid = use_musdb_test_as_valid
        self.duration = duration
        self.batch_size = batch_size
        self.train_mono = train_mono
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = int(np.ceil(sr * duration / hop_length))
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if self.format=="MSS":
            if stage == 'fit' or stage == 'validate' or stage is None:
                self.train_set = MSSSpecs(data_dir=self.base_dir, samples_per_track=self.samples_per_track, subset='train',
                                          dataset_str=self.dataset_str, valid_split = self.valid_split, rand_seed=self.rand_seed,
                                          target_str=self.target_str, random_mix_flag=self.random_mix, 
                                          augmentation_flag=self.aug_flag, enforce_full_mix_percentage=self.full_mix_percentage, duration=self.duration, dummy=self.dummy, 
                                          normalize=self.normalize, train_mono=self.train_mono, **specs_kwargs)
                if self.use_musdb_test_as_valid:
                    self.valid_set = MSSSpecs(data_dir=self.base_dir, samples_per_track=self.samples_per_track, subset='test',
                                          dataset_str=self.dataset_str, valid_split = 0.0, rand_seed=self.rand_seed,
                                          target_str=self.target_str, random_mix_flag=False, 
                                          augmentation_flag=False, enforce_full_mix_percentage=0, duration=self.duration, dummy=self.dummy, 
                                          normalize=self.normalize, train_mono=self.train_mono, **specs_kwargs)
                else:
                    self.valid_set = MSSSpecs(data_dir=self.base_dir, samples_per_track=self.samples_per_track, subset='valid',
                                              dataset_str=self.dataset_str, valid_split = self.valid_split, rand_seed=self.rand_seed,
                                              target_str=self.target_str, random_mix_flag=False, 
                                              augmentation_flag=False, enforce_full_mix_percentage=1, duration=self.duration, dummy=self.dummy, 
                                              normalize=self.normalize, train_mono=self.train_mono, **specs_kwargs)
            
            if stage == 'test' or stage is None:
                self.test_set = MSSSpecs(data_dir=self.base_dir, samples_per_track=self.samples_per_track, subset='test',
                                          dataset_str=self.dataset_str, valid_split = self.valid_split, rand_seed=self.rand_seed,
                                          target_str=self.target_str, random_mix_flag=False, 
                                          augmentation_flag=False, enforce_full_mix_percentage=1, duration=self.duration, dummy=self.dummy, 
                                          normalize=self.normalize, train_mono=self.train_mono, **specs_kwargs)
        else:
            if stage == 'fit' or stage =='validate' or stage is None:
                self.train_set = Specs(data_dir=self.base_dir, subset='train',
                    dummy=self.dummy, shuffle_spec=True, format=self.format,
                    normalize=self.normalize, **specs_kwargs)
                self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
                    dummy=self.dummy, shuffle_spec=False, format=self.format,
                    normalize=self.normalize, **specs_kwargs)
            if stage == 'test' or stage is None:
                self.test_set = Specs(data_dir=self.base_dir, subset='test',
                    dummy=self.dummy, shuffle_spec=False, format=self.format,
                    normalize=self.normalize, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True, prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
