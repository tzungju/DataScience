import flask 

app = flask.Flask(__name__)

@app.route("/", methods=['GET', "POST"])
def hello():
    return "Hello World!"
       
app.run()

# 從 PC Home copy 一段 Html
