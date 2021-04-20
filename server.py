from flask import Flask, request
from utils import TMP_DIR, get_room_posibilities
from pathlib import Path
from uuid import uuid4

app = Flask("WPI backend")

@app.route('/', methods=["POST"])
def determine_room_route():
	if 'target' not in request.files:
		return "No file part.", 400

	target = request.files['target']
	if target.filename == '':
		return "No target photo passed.", 400

	Path(TMP_DIR).mkdir(exist_ok=True)
	target.filename = str(uuid4()) + '.jpg'
	target.save(TMP_DIR)

	room_posibilities = get_room_posibilities(target.filename)
    #TODO: delete from TMP
	return room_posibilities, 200

if __name__ == '__main__':
	app.run()