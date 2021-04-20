from flask import Flask


app = Flask("WPI backend")

@app.route('/', methods=["GET"])
def myroute():
	# TODO process photo
	return {'message': "hello world!"}, 200

if __name__ == '__main__':
	app.run()