from flask import Flask, request
from utils import save_target_image, RoomDeterminant

app = Flask("WPI backend")


@app.route('/', methods=["POST"])
def determine_room_route():
    if 'target' not in request.files:
        return "No file part.", 400

    target = request.files['target']
    if target.filename == '':
        return "No target photo passed.", 400

    print(target, type(target))
    save_target_image(target)

    determinant = RoomDeterminant()
    room_possibilities = determinant.get_room_possibilities(target.filename)
    return room_possibilities, 200


if __name__ == '__main__':
    app.run(use_reloader=True)
