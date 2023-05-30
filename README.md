# RegGPT

A machine learning model that automatically fits the best regression line to any given dataset whose entries are of the form `(x, y)`.

<p align = "center">
    <img src = "./assets/slr.gif", alt = "SLR animation", height = 300px>
</p>

## Dependencies

The build for this program can be entirely done by running one of the scripting files suitable for your operating system and terminal of choice. If you are using the Windows CMD terminal then run the `.\build.bat` file while running the `.\build.ps1` file for Windows powershell. For any WLS or Linux based terminal system run the following command,

```bash
source ./scripts/build.sh
```
If you are unable to run the above command due any permission issues then run the command `chmod +x ./scripts/build.sh` to give the `build.sh` file the necessary permissions to be executed.