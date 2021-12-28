from rich.table import Table
from rich.console import Console
from pytorch_lightning import Callback


class RichDataSummary(Callback):
    def on_pretrain_routine_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero:
            console = Console()
            data = trainer.datamodule
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Dataset Name", justify="center")
            table.add_column("Num Class", justify="center")
            table.add_column("Batch Size", justify="right")
            table.add_column("Step", justify="right")
            table.add_column("Train", justify="right")
            table.add_column("Test", justify="right")
            table.add_row(
                data.dataset_name,
                str(data.num_classes),
                str(data.batch_size),
                str(data.num_step),
                str(data.train_data_len),
                str(data.test_data_len)
            )
            console.print(table)