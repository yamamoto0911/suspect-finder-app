#!/usr/bin/env python3
"""
DumpWatcher - 不法投棄検知システム
使用方法: python main.py <入力動画ファイル> [出力動画ファイル]
"""

import sys
import os
from dump_watcher import DumpWatcher

def main():
    # コマンドライン引数のチェック
    if len(sys.argv) < 2:
        print("使用方法: python main.py <入力動画ファイル> [出力動画ファイル]")
        print("例: python main.py input_video.mp4")
        print("例: python main.py input_video.mp4 output_result.mp4")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
    
    # 入力ファイルの存在確認
    if not os.path.exists(input_video):
        print(f"エラー: 入力ファイル '{input_video}' が見つかりません。")
        sys.exit(1)
    
    print("=" * 60)
    print("DumpWatcher - 不法投棄検知システム")
    print("=" * 60)
    print(f"入力動画: {input_video}")
    print(f"出力動画: {output_video}")
    print(f"証拠画像: evidence_*.jpg")
    print(f"ログファイル: dump_log.json")
    print("=" * 60)
    
    try:
        # DumpWatcherを初期化
        print("YOLOモデルを読み込み中...")
        watcher = DumpWatcher()
        
        # 動画を処理
        print("動画解析を開始します...")
        watcher.process_video(input_video, output_video)
        
        print("=" * 60)
        print("解析完了!")
        print(f"確定した不法投棄イベント: {len(watcher.confirmed_events)}件")
        print(f"保留中のイベント: {len([e for e in watcher.dumping_events if not e.confirmed])}件")
        
        if watcher.confirmed_events:
            print("\n検知された不法投棄:")
            for i, event in enumerate(watcher.confirmed_events, 1):
                print(f"  {i}. フレーム{event.frame_number}: "
                      f"{event.suspect_class}が{event.object_class}を投棄")
        
        print(f"\n出力ファイル:")
        print(f"  - 結果動画: {output_video}")
        print(f"  - ログファイル: dump_log.json")
        if watcher.evidence_count > 0:
            print(f"  - 証拠画像: evidence_*.jpg ({watcher.evidence_count}枚)")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()