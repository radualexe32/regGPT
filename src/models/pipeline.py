from classificationGPT import *
from regGPT import *
from statisticsGPT import *
from flask import Flask
import gradio as gr
import pandas as pd


def nested(file):
    df = pd.read_csv(file)

    nested_list = []
    for index, row in df.iterrows():
        curr_row = []
        for column in df.columns:
            curr_row.append(row[column])
        nested_list.append(curr_row)
    return nested_list


def data_classification(file, text, number):
    data = nested(file.name)
    reg_types = text.split(", ")
    out = classifier(data=data, correlation=number, reg_types=reg_types)
    return out


iface = gr.Interface(
    fn=data_classification, inputs=["file", "text", "number"], outputs="text", title="🚀 regGPT")


# app = Flask(__name__)
#
#

# @app.route('/')
# def index():
#    # Get user inputs
#    return '<p>Hello world!</p>'
#
#
# def main():
#    print("Pipeline for linking models together")
#
#


if __name__ == "__main__":
    # main()
    iface.launch()
