from proto.test_pb2 import Location
from websockets.server import serve
import asyncio
import argparse

async def echo(websocket):
    async for message in websocket:
        location = Location()
        location.ParseFromString(message)
        location.x += 1
        location.y -= 1
        print(location)
        await websocket.send(location.SerializeToString())

async def main():
    parser = argparse.ArgumentParser(prog="RNN ONNX Server")
    parser.add_argument("--ip", required=False, default="localhost")
    parser.add_argument("--port", required=False, default=8765)
    args = parser.parse_args()
    
    async with serve(echo, args.ip, args.port):
        print(f"Running @ {args.ip}:{args.port}")
        await asyncio.Future()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass