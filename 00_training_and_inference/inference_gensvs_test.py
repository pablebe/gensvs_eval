from gensvs import MelRoFoBigVGAN, SGMSVS

MIX_PATH = '/home/bereuter/MSS_experiments/gensvs_eval/03_audio_examples/Mixture'
SEP_PATH = '/home/bereuter/MSS_experiments/gensvs_eval/03_audio_examples/'
sgmsvs_model = SGMSVS() 
melrofo_model = MelRoFoBigVGAN()

sgmsvs_model.run_folder(MIX_PATH, SEP_PATH, loudness_normalize=True, loudness_level=-18, output_mono=True)
melrofo_model.run_folder(MIX_PATH, SEP_PATH, loudness_normalize=True, loudness_level=-18, output_mono=True)


print("stop")