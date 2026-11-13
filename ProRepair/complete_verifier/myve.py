import yaml

def update_yaml_path(file_path, new_model_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # 更新路径
    config['model']['path'] = new_model_path

    # 将更新后的内容写回文件
    with open(file_path, 'w') as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    from abcrown import ABCROWN
    import sys
    yaml_file = 'exp_configs/myverify.yaml'
    method = 'mag'
    method = 'refine_score'
    for i in range(2, 6):
        for j in range(1, 10):
            new_path = f'/data/home/mjnn/majianan/ProvRepair/result/global_safety/Ours_{method}/n{i}{j}_repair.pth'
            update_yaml_path(yaml_file, new_path)
            abcrown = ABCROWN(args=sys.argv[1:])
            result = abcrown.main()
            log_path = f"/data/home/mjnn/majianan/ProvRepair/result/global_safety/Ours_{method}/verify.log"
            log_file = open(log_path, 'a')
            log_info = []
            if 'unsafe-pgd' in result.keys():
                log_info.append(f"Repair net n{i}{j} Failed \n")
            elif 'safe' in result.keys():
                log_info.append(f"Repair net n{i}{j} Success \n")
            log_file.write(''.join(log_info))
            log_file.flush()
