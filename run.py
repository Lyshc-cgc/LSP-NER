from module.annotation import Annotation
from module.processor import Processor
from module.func_util import get_config

def main():
    config = get_config('config.yml')
    # 1. pre-process the data
    proc = Processor(config['data_cfg'])
    formated_dataset = proc.process()

    # 2. annotate the data by LLMs
    prompt_type = 'few_shot_prompt'
    assert prompt_type in ('raw_prompt', 'single_type_prompt', 'multi_type_prompt', 'few_shot_prompt')
    for anno_cfg in config['anno_cfgs'][prompt_type]:
        anno = Annotation(anno_cfg)
        anno.annotate_by_all(formated_dataset, quality=False)

if __name__ == '__main__':
    main()