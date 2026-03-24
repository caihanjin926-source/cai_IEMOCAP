import os
import pandas as pd

# ================== 需要修改的路径 ==================
# 1. 指向你电脑上 IEMOCAP 的主文件夹 (请务必确保这个路径下直接包含 Session1, Session2 等文件夹)
# 如果你解压后多了一层，可能需要改成 r"D:\IEMOCAP\IEMOCAP_full_release" 之类的
iemocap_root = r"D:\mobile_audio\DataSet\IEMOCAP_full_release"
# 2. 生成的全新 6分类 CSV 保存位置
save_csv_path = r"D:\mobile_audio\DataSet\IEMOCAP_Labels\my_iemocap_labels_6class.csv"
# ====================================================

target_emotions = ['ang', 'hap', 'exc', 'sad', 'neu', 'fru']
data = []

print("开始解析 IEMOCAP 原始标签文件 (6 分类)...")

for session in range(1, 6):
    eval_dir = os.path.join(iemocap_root, f'Session{session}', 'dialog', 'EmoEvaluation')

    if not os.path.exists(eval_dir):
        print(f"警告: 找不到路径 {eval_dir}，请检查 iemocap_root 路径是否正确！")
        continue

    for file in os.listdir(eval_dir):
        if not file.endswith('.txt'):
            continue

        file_path = os.path.join(eval_dir, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('['):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    filename = parts[1]
                    emotion = parts[2]

                    if emotion in target_emotions:
                        subject = filename[:6]
                        data.append({
                            'Filename': filename,
                            'Emotion': emotion,
                            'Subject': subject
                        })

df = pd.DataFrame(data)

# 新增：防崩溃检查
if len(df) == 0:
    print("\n❌ 错误：没有提取到任何数据！")
    print("👉 请检查第7行的 iemocap_root 路径，确保该路径下能直接看到 Session1 文件夹。")
else:
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df.to_csv(save_csv_path, index=False)
    print(f"\n✅ 标签提取成功！")
    print(f"共提取了 {len(df)} 条有效的 6 分类语音标签。")
    print(f"各类别数量统计:\n{df['Emotion'].value_counts()}")
    print(f"文件已保存在: {save_csv_path}")