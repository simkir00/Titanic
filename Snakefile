from pathlib import Path

# ToDo Разнести на этапы логичнее (и аккуратнее)

# rule prepare_data:
#    params:
#        cli=Path("workflow/pipeline.py")
#    shell:
#        "python {params.cli} -t prepare_data"

# rule create_features:
#    params:
#        cli=Path("workflow/pipeline.py")
#    shell:
#        "python {params.cli} -t full"

# rule make_predictions

rule full pipeline:
    params:
        cli=Path("workflow/pipeline.py")
    shell:
        "python {params.cli} -t full"