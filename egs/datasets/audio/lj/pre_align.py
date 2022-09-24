from data_gen.tts.base_preprocess import BasePreprocessor


class LJPreAlign(BasePreprocessor):
    def meta_data(self):
        for l in open(f'{self.raw_data_dir}/metadata.csv').readlines():
            item_name, _, txt = l.strip().split("|")
            wav_fn = f"{self.raw_data_dir}/wavs/{item_name}.wav"
            # yield item_name, wav_fn, txt, 'SPK1'
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': 'SPK1', 'txt_loader': None}



if __name__ == "__main__":
    LJPreAlign().process()
