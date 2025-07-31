from gensvs import MelRoFoBigVGAN, SGMSVS

#WORKERS = 
MIX_PATH = './audio_examples/mixture'
SEP_PATH = './audio_examples/separated'


sgmsvs_model = SGMSVS()
melrofo_model = MelRoFoBigVGAN()

sgmsvs_model.run_folder(MIX_PATH, SEP_PATH, loudness_normalize=False, loudness_level=-18, output_mono=True)
melrofo_model.run_folder(MIX_PATH, SEP_PATH, loudness_normalize=False, loudness_level=-18, output_mono=True)

