#!/bin/bash
# Proto 파일 컴파일 스크립트
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Compiling proto files..."
uv run python -m grpc_tools.protoc \
    -I"$PROJECT_ROOT/proto" \
    --python_out="$PROJECT_ROOT/proto" \
    --grpc_python_out="$PROJECT_ROOT/proto" \
    "$PROJECT_ROOT/proto/video_service.proto"

# Fix import path in generated grpc file
sed -i 's/import video_service_pb2/from proto import video_service_pb2/' \
    "$PROJECT_ROOT/proto/video_service_pb2_grpc.py"

echo "Proto compilation complete."
echo "Generated files:"
ls -la "$PROJECT_ROOT/proto/"*.py
