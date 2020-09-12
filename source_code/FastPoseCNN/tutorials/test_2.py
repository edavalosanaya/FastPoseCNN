from catalyst.dl import SupervisedRunner, AlchemyLogger

runner = SupervisedRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,asdf
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True,
    callbacks={
        "alchemy_logger": AlchemyLogger(
            token="16511fcaebab915c1f3a00fea3b1485e", # your Alchemy token
            project="default",
            experiment="your_experiment_name",
            group="your_experiment_group_name",
        )
    }
)