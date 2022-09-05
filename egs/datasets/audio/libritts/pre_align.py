import os

from data_gen.tts.base_pre_align import BasePreAlign
import glob


class LibrittsPreAlign(BasePreAlign):
    def meta_data(self):
        wav_fns = sorted(glob.glob(f'{self.raw_data_dir}/*/*/*/*.wav'))
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            txt_fn = f'{wav_fn[:-4]}.normalized.txt'
            spk = item_name.split("_")[0]
            yield item_name, wav_fn, (self.load_txt, txt_fn), spk


if __name__ == "__main__":
    LibrittsPreAlign().process()
