from imports import *
from classificationGPT import *
from regGPT import *
from dataset import *
from embeddings import *
from statisticsGPT import *

parser = argparse.ArgumentParser(
    description='Use regGPT to predict values for a given dataset')
parser.add_argument('--data', type=str, required=True,
                    help='Path to CSV file containing dataset')
args = parser.parse_args()


class OutputFlaggingCallback(gr.FlaggingCallback):
    def __init__(self):
        self.output = []


def nested(file):
    df = pd.read_csv(file)

    nested_list = []
    for _, row in df.iterrows():
        curr_row = []
        for column in df.columns:
            curr_row.append(row[column])
        nested_list.append(curr_row)

    return nested_list


# def data_format(file):
#     df = pd.read_csv(file)
#     cols = df.columns.tolist()
#     dep_col = cols[0]
#     ind_col = cols[1]

#     x = df[ind_col].values.tolist()
#     y = df[dep_col].values.tolist()

#     X_tensor = torch.tensor(x, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

#     return X_tensor, y_tensor


def components_link(file, reg_types, number, text):
    # Functional calls for linking all components together.
    # 1. Preprocessing
    # data_embedding = CSVEmbedding(file.name)
    if isinstance(file, str):
        data = nested(file)
    else:
        data = nested(file.name)

    # 2. Classifcation
    out_class = classifier(
        data=data, reg_types=reg_types, correlation=number, extra=text
    )

    # 3. Reg line computation
    reg = Regression(
        input_dim=1,
        output_dim=1,
        regression_type=out_class.reg_type,
        degree=int(out_class.deg_range),
    )

    if isinstance(file, str):
        out_reg = reg_out(file, mod=reg)
    else:
        out_reg = reg_out(file.name, mod=reg)

    # 4. Stats suggestions
    classifier_dict = {
        "ind_var": out_class.ind_var,
        "dep_var": out_class.dep_var,
    }
    out_stats = inference_generator(
        variables=classifier_dict, reg_model=out_reg)

    stats_reg = {
        "MSE": out_reg.get_mse(),
        "R2": out_reg.get_r2(),
        "Correlation": out_reg.get_correlation(),
    }

    class_str = """
    Classification model:
    {out_class}

    Regression model:
    {out_reg}

    Relevant stats:
    {stats_reg}

    Statistics suggestions:
    {out_stats}
    """
    return class_str.format(
        out_class=out_class,
        out_reg=out_reg.get(),
        stats_reg=stats_reg,
        out_stats=out_stats,
    )


def reg_out(file, mod=Regression()):
    # Create data loaders for training and validation
    train_data = RegDataset(file)
    train_loader = train_data.get_dataloader()

    val_data = RegDataset(file, train=False)
    val_loader = val_data.get_dataloader()

    # Train model
    mod.train(train_loader, val_loader, epochs=1000)
    return mod


def cli():
    correlation_coefficient = input(
        "Enter the correlation coefficient (default: 0.5): ")
    if correlation_coefficient == '':
        correlation_coefficient = 0.5

    regression_type = input(
        "Choose the type of regression (1-simple, 2-multi, 3-polynomial, 4-logistic): ")

    out = components_link(args.data, regression_type,
                          correlation_coefficient, "")
    print(out)
    # TODO: Start of personalized AI chatbot


def gradio_interface():
    # callback = OutputFlaggingCallback()
    iface = gr.Interface(
        fn=components_link,
        inputs=[
            gr.components.File(label="Data File"),
            gr.components.CheckboxGroup(
                ["simple", "multi", "polynomial", "logistic"], label="Regression Types"
            ),
            gr.components.Slider(
                minimum=-1, maximum=1, step=0.0001, label="Correlation"
            ),
            gr.components.Textbox(label="Extra Information"),
        ],
        outputs="text",
        flagging_callback=gr.SimpleCSVLogger(),
        title="🚀 regGPT",
        description="RegGPT is a tool that helps you find the best regression model given some dataset and gives you suggestions on the types of tests that would elevate your statistical analysis. Before you begin, please make sure that your data is in a CSV file format, have a correlation coefficient handy and a general idea of what regression types you should use. Of course the last case is a bit special since you can always check all of the boxes if you are not sure. But if you want a more specific query, any extra information you can give to model will help it give you a better answer.",
    )
    iface.launch()


if __name__ == "__main__":
    # gradio_interface()
    cli()
