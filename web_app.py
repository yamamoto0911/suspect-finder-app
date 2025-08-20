from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, flash
import os
import uuid
import json
from werkzeug.utils import secure_filename
from dump_watcher import DumpWatcher
import threading
import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'dumpwatcher_secret_key_2024'

# 設定
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB制限

# ディレクトリ作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 解析状況を管理するグローバル変数
analysis_status = {}

def allowed_file(filename):
    """許可されたファイル形式かチェック"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_video_async(task_id, input_path, output_dir):
    """非同期で動画解析を実行"""
    try:
        analysis_status[task_id]['status'] = 'processing'
        analysis_status[task_id]['message'] = 'YOLOモデルを読み込み中...'
        
        # DumpWatcherを初期化
        watcher = DumpWatcher()
        
        analysis_status[task_id]['message'] = '動画解析を開始中...'
        
        # 出力ファイルパス
        output_video = os.path.join(output_dir, 'output.mp4')
        
        # 元のカレントディレクトリを保存
        original_dir = os.getcwd()
        
        # 絶対パスに変換（ディレクトリ移動前に実行）
        abs_input_path = os.path.abspath(input_path)
        abs_output_video = os.path.abspath(output_video)
        print(f"デバッグ: 元ディレクトリ = {original_dir}")
        print(f"デバッグ: 入力パス = {abs_input_path}")
        print(f"デバッグ: 出力パス = {abs_output_video}")
        print(f"デバッグ: 入力ファイル存在確認 = {os.path.exists(abs_input_path)}")
        
        try:
            # 出力ディレクトリに移動（証拠画像の保存場所を制御）
            os.chdir(output_dir)
            print(f"デバッグ: 移動後ディレクトリ = {os.getcwd()}")
            
            # シンプルな動画処理を実行
            watcher.process_video_simple(abs_input_path, abs_output_video)
            
        except Exception as e:
            print(f"動画処理中にエラーが発生: {str(e)}")
            # エラーの場合でも基本情報は返す
        finally:
            # 元のディレクトリに戻る
            os.chdir(original_dir)
        
        # 証拠画像の数を実際のファイル数から取得
        evidence_count = 0
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.startswith('evidence_') and file.endswith('.jpg'):
                    evidence_count += 1
        
        # 結果をまとめる
        results = {
            'confirmed_events': len(watcher.confirmed_events),
            'pending_events': len([e for e in watcher.dumping_events if not e.confirmed]),
            'evidence_images': evidence_count,
            'output_video': output_video if os.path.exists(output_video) else None,
            'total_frames_processed': watcher.current_frame,
            'video_generated': os.path.exists(output_video),
            'events_detail': [
                {
                    'frame': event.frame_number,
                    'time_in_video': watcher.frame_to_time(event.frame_number),
                    'suspect': event.suspect_class,
                    'object': event.object_class,
                    'timestamp': event.timestamp,
                    'distance_change': f"{event.initial_distance:.1f}→{event.final_distance:.1f}px"
                }
                for event in watcher.confirmed_events
            ]
        }
        
        analysis_status[task_id]['status'] = 'completed'
        analysis_status[task_id]['message'] = '解析完了'
        analysis_status[task_id]['results'] = results
        
    except Exception as e:
        analysis_status[task_id]['status'] = 'error'
        analysis_status[task_id]['message'] = f'エラーが発生しました: {str(e)}'

@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """ファイルアップロードと解析開始"""
    if 'file' not in request.files:
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # 一意のタスクIDを生成
        task_id = str(uuid.uuid4())
        
        # ファイル名を安全にする
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        
        # ファイルを保存
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # 結果保存用ディレクトリを作成
        output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 解析状況を初期化
        analysis_status[task_id] = {
            'status': 'queued',
            'message': '解析待機中...',
            'filename': filename,
            'start_time': datetime.now().isoformat()
        }
        
        # 非同期で解析を開始
        thread = threading.Thread(
            target=analyze_video_async,
            args=(task_id, input_path, output_dir)
        )
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('results', task_id=task_id))
    
    else:
        flash('対応していないファイル形式です。MP4, AVI, MOV, MKVファイルをアップロードしてください。')
        return redirect(request.url)

@app.route('/results/<task_id>')
def results(task_id):
    """結果表示ページ"""
    if task_id not in analysis_status:
        flash('無効なタスクIDです')
        return redirect(url_for('index'))
    
    return render_template('results.html', task_id=task_id)

@app.route('/api/status/<task_id>')
def get_status(task_id):
    """解析状況をAPI経由で取得"""
    if task_id not in analysis_status:
        return jsonify({'error': '無効なタスクID'}), 404
    
    return jsonify(analysis_status[task_id])

@app.route('/download/<task_id>/<file_type>')
def download_file(task_id, file_type):
    """結果ファイルをダウンロード"""
    if task_id not in analysis_status:
        return jsonify({'error': '無効なタスクID'}), 404
    
    if analysis_status[task_id]['status'] != 'completed':
        return jsonify({'error': '解析が完了していません'}), 400
    
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    
    if file_type == 'video':
        file_path = os.path.join(output_dir, 'output.mp4')
        if not os.path.exists(file_path):
            # さらに詳細なエラー情報を提供
            available_files = []
            if os.path.exists(output_dir):
                available_files = os.listdir(output_dir)
            return jsonify({
                'error': '動画ファイルが見つかりません',
                'details': f'パス: {file_path}',
                'available_files': available_files,
                'directory_exists': os.path.exists(output_dir)
            }), 404
        return send_file(file_path, as_attachment=True, download_name='dump_detection_result.mp4')
    
    elif file_type == 'evidence':
        # 証拠画像の一覧を取得
        evidence_files = []
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.startswith('evidence_') and file.endswith('.jpg'):
                    evidence_files.append(file)
        
        if not evidence_files:
            return jsonify({'error': '証拠画像が見つかりません'}), 404
        
        # 最初の証拠画像を返す
        evidence_path = os.path.join(output_dir, evidence_files[0])
        return send_file(evidence_path, as_attachment=False, mimetype='image/jpeg')
    
    else:
        return jsonify({'error': '無効なファイルタイプ'}), 400

@app.route('/api/cleanup/<task_id>', methods=['POST'])
def cleanup_task(task_id):
    """タスクのクリーンアップ"""
    if task_id not in analysis_status:
        return jsonify({'error': '無効なタスクID'}), 404
    
    try:
        # アップロードファイルを削除
        output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        
        # ステータスから削除
        del analysis_status[task_id]
        
        return jsonify({'message': 'クリーンアップ完了'})
    
    except Exception as e:
        return jsonify({'error': f'クリーンアップエラー: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)