from module.annotation import Annotation
from module.processor import Processor
from module.func_util import get_config

def main():
    config = get_config('config.yml')
    # 1. pre-process the data
    proc = Processor(config['data_cfg'])
    formated_dataset = proc.process()

    # 2. annotate the data by LLMs
    use_api = True
    prompt_type = 'st_few_shot'
    api_cfg = config['api_cfg'] if use_api else None
    assert prompt_type in ('raw', 'single_type', 'multi_type', 'few_shot', 'st_few_shot')
    for anno_cfg in config['anno_cfgs'][prompt_type]:
        anno = Annotation(anno_cfg, api_cfg)
        anno.annotate_by_all(formated_dataset, quality=False)

if __name__ == '__main__':
    main()