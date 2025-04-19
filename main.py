import argparse
from matcher import match_and_display_image, match_and_save_video

parser = argparse.ArgumentParser(description="Feature Matching using SIFT + FLANN")
parser.add_argument('--mode', type=str, choices=['image', 'video'], required=True)
parser.add_argument('--query', type=str, help='Path to query image')
parser.add_argument('--target', type=str, help='Path to target image (or video)')
args = parser.parse_args()

if args.mode == 'image':
    match_and_display_image(args.query, args.target)
elif args.mode == 'video':
    match_and_save_video(args.query, args.target)
