from flask import Flask
app=Flask(__name__)

@app.route('/')
def hello():
    return "Hello World from Joie!"

@app.route('/user')
def hello_name():
    return "hello user!"

if __name__ == "__main__":
    app.run()
