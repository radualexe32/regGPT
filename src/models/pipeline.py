from classificationGPT import *
from regGPT import *
from statisticsGPT import *
from flask import Flask
import gradio as gr
import pandas as pd
from gradio import themes
import torch


def nested(file):
    df = pd.read_csv(file)

    nested_list = []
    for _, row in df.iterrows():
        curr_row = []
        for column in df.columns:
            curr_row.append(row[column])
        nested_list.append(curr_row)

    return nested_list


def data_format(file):
    df = pd.read_csv(file)
    cols = df.columns.tolist()
    dep_col = cols[0]
    ind_col = cols[1]

    x = df[ind_col].values.tolist()
    y = df[dep_col].values.tolist()

    X_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor


def data_classification(file, reg_types, number, text):
    data = nested(file.name)
    out = classifier(data=data, correlation=number,
                     reg_types=reg_types, extra=text)

    class_str = """
    The following information is useful for your regression analysis. Does this sound right?

    Independent Variable: {ind_var}
    Dependent Variable: {dep_var}
    Regression Type: {reg_type}
    Degree Range: {deg_range}
    """
    return class_str.format(ind_var=out.ind_var, dep_var=out.dep_var, reg_type=out.reg_type, deg_range=out.deg_range)


def reg_out():
    raise NotImplementedError


def stats_suggestions():
    raise NotImplementedError


def gradio_interface():
    iface = gr.Interface(
        fn=data_classification,
        inputs=[
            gr.components.File(label="Data File"),
            gr.components.CheckboxGroup(
                ["linear", "polynomial", "logistic"], label="Regression Types"),
            gr.components.Slider(minimum=-1, maximum=1,
                                 step=0.0001, label="Correlation"),
            gr.components.Textbox(label="Extra Information")
        ],
        outputs="text",
        title="🚀 regGPT",
        description=".RegGPT is a tool that helps you find the best regression model given some dataset and gives you suggestions on the types of tests that would elevate your statistical analysis. Before you begin, please make sure that your data is in a CSV file format, have a correlation coefficient handy and a general idea of what regression types you should use. Of course the last case is a bit special since you can always check all of the boxes if you are not sure. But if you want a more specific query, any extra information you can give to model will help it give you a better answer."
    )
    iface.launch()


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
    gradio_interface()
