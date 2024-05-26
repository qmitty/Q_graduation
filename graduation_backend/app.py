from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
import csv
import os
import zipfile
import subprocess
app = Flask(__name__)
CORS(app)  # 启用 CORS
# 设置dataset文件夹的路径
DATASET_FOLDER = os.path.join(os.getcwd(), 'dataset')

# 确保dataset文件夹存在
os.makedirs(DATASET_FOLDER, exist_ok=True)
# 配置静态文件夹
app.config['SEGIMAGE_FOLDER'] = 'outcome/segimage'

# 创建路由
@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['SEGIMAGE_FOLDER'], filename)
@app.route('/hello', methods=['POST'])
def hello():
    # 解析请求中的JSON数据以获取文件名
    filename = request.json.get('filename')
    if not filename:
        return jsonify({'error': '文件名为空'}), 400

    # 构建文件的完整路径
    file_path = os.path.join(os.getcwd(), 'dataset', filename)

    # 检查文件是否存在
    if not os.path.isfile(file_path):
        return jsonify({'error': '文件不存在'}), 404

    data = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            epoch = row['Epoch'] if row['Epoch'] else 0
            loss = float(row['Loss']) if row['Loss'] else 0.0
            f_score = float(row['F-Score']) if row['F-Score'] else 0.0
            mcc = float(row['MCC']) if row['MCC'] else 0.0
            
            data.append({
                'Epoch': epoch,
                'Loss': loss,
                'F-Score': f_score,
                'MCC': mcc
            })

    return jsonify(data)

@app.route('/files', methods=['GET'])
def list_files():
    # 假设您的文件存储在项目的dataset目录下
    directory = os.path.join(os.getcwd(), 'dataset')
    # 只获取以.csv结尾的文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.csv')]
    return jsonify({'files': files})

@app.route('/folders', methods=['GET'])
def list_folders():
    # 假设您的文件夹存储在项目的dataset目录下
    directory = os.path.join(os.getcwd(), 'dataset')
    # 获取所有的文件夹
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return jsonify({'folders': folders})

@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查请求中是否有文件部分
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    file = request.files['file']
    # 检查是否选择了文件
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    # 保存文件
    if file:
        filename = file.filename  # 直接使用上传的文件名
        save_path = os.path.join(os.getcwd(), './dataset', filename)
        file.save(save_path)
        return jsonify({'message': '文件上传成功', 'path': save_path}), 200
    
@app.route('/upload-zip', methods=['POST'])
def upload_zip_file():
    # 检查请求中是否有文件部分
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    file = request.files['file']
    # 检查是否选择了文件
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    if not file.filename.endswith('.zip'):
        return jsonify({'error': '文件不是zip格式'}), 400
    # 保存文件
    filename = file.filename  # 直接使用上传的文件名
    save_path = os.path.join(DATASET_FOLDER, filename)
    file.save(save_path)
    return jsonify({'message': 'zip文件上传成功', 'path': save_path}), 200

@app.route('/unzip-datasets', methods=['GET'])
def unzip_datasets():
    for item in os.listdir(DATASET_FOLDER):
        if item.endswith('.zip'):
            zip_path = os.path.join(DATASET_FOLDER, item)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATASET_FOLDER)
            os.remove(zip_path)
    return jsonify({'message': '所有zip文件已解压并删除'}), 200
@app.route('/train', methods=['POST'])
def train():
    # 打印接收到的所有参数
    print(request.json)
    # 从请求中获取数据集参数
    dataset = request.json.get('selectedFolder', 'ABNORMAL')
    num_classes = request.json.get('num_classes', '2')
    init_epoch = request.json.get('Init_Epoch', '0')
    save_period = request.json.get('save_period', '5')
    pretrained = request.json.get('pretrained', 'True')
    model_path = request.json.get('model_path', '')
    freeze_train = request.json.get('Freeze_Train', 'False')
    freeze_epoch = request.json.get('Freeze_Epoch', '0')
    freeze_batch_size = request.json.get('Freeze_batch_size', '2')
    unfreeze_epoch = request.json.get('UnFreeze_Epoch', '50')
    unfreeze_batch_size = request.json.get('Unfreeze_batch_size', '2')
    # 构建运行脚本的命令，确保路径使用正斜杠，并且不添加额外的引号
    command = f"python train_medical2.py --dataset ./dataset/{dataset}"

    # 从请求中获取其他参数并添加到命令中


    # 组装命令
    command += f" --num_classes {num_classes} --Init_Epoch {init_epoch} --save_period {save_period}"
    command += f" --pretrained {pretrained} "
    command += f" --Freeze_Train {freeze_train} --Freeze_Epoch {freeze_epoch} --Freeze_batch_size {freeze_batch_size}"
    command += f" --UnFreeze_Epoch {unfreeze_epoch} --Unfreeze_batch_size {unfreeze_batch_size}"

    try:
        print(command)
        # 运行命令
        subprocess.run(command, shell=True, check=True)
        return jsonify({'message': 'Training started successfully'}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Failed to start training', 'details': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
