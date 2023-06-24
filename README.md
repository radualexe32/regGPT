# 🚀 regGPT 

Statistics tool that combines various models into one pipeline to output the best regression line for any given dataset and various hypothesis tests, predictive tools, etc. for further data analysis. Pipeline consists of the following models:
- `classficationGPT` is the model that classifies the data into either linear, logistic or polynomial regression, with a correct degree range and the actual data types from the user inputted data.
- `regGPT` the singular model that applies the correct number of linear layers and activation functions for the specified regression model that is outputted from the `classificationGPT` model.
- `statisticsGPT` the model that outputs the further analysis tools based on the outputs of the previous two models.

## Dependencies

In the `scripts/` directory give permissions to the `build.sh` file with `chmod +x scripts/build.sh` while in the root directory and set up the virtual environment with the following code,

```bash
source scripts/build.sh
```

### API Keys

To use your own [OpenAI](https://platform.openai.com/overview) API key run the following command while in the root directory of the project.

```bash
python3 source/api_key.py --openai <your_api_key>
```

As of June 13, 2023, the API key does not work for using any GPT-4 model (none will be called anyway). In the future make sure that you have the necessary permissions to run the GPT-4 model.

## Gradio UI

The UI for this project is built using [Gradio](https://gradio.app/). The current state is very basic. Run the following command to run the app locally.

```bash
python3 src/models/pipeline.py
```

Make sure to have a couple of things at hand. Some sort of `*.csv` file, and the correlation coefficient of the data.