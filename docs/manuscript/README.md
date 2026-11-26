# Ant foraging reproduction manuscript

This software companion documents the existing numerical reproduction.
The published study has its own [journal DOI](https://doi.org/10.1016/j.cogsys.2023.02.005).
The numbered sections, configuration, preamble, bibliography, and
[comparison figure assets](assets/) live here.

To regenerate the figure under the [pinned numerical environment](../README.md#setup),
run this command from the repository root.

```bash
.venv/bin/python docs/manuscript/assets/render_figure.py docs/manuscript/assets/foraging-curves.png
```

## Render and validate

Use a clean source checkout and a separate clone of the
[pinned template renderer](https://github.com/docxology/template/tree/87704feae0a4490ef7dad9c931d268e8c30273c2).
Install Git, uv, Pandoc, XeLaTeX, and the TeX packages listed in the template
[setup guide](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/guides/startup-and-setup.md).
The PDF build uses the checked-in figure and registry.

Run from the root of the ant-pheromone checkout.

```bash
model_checkout="$(pwd -P)"
renderer_checkout="$(mktemp -d)/template"
mkdir -p output/figures
cp docs/manuscript/assets/foraging-curves.png docs/manuscript/assets/figure_registry.json output/figures/

git clone https://github.com/docxology/template.git "$renderer_checkout"
git -C "$renderer_checkout" checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
cd "$renderer_checkout"
uv sync --locked
mkdir -p projects/working
ln -s "$model_checkout" projects/working/ant-pheromone
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/ant-pheromone
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/ant-pheromone
```

Read `output/pdf/ant-pheromone_combined.pdf` in the model checkout. Check
the experiment, methods, curves, limits, and references sections, including
both panels in the figure. Review `output/reports/validation_report.md` and
`output/reports/rendered_provenance.json` for the validation verdict and source
bindings.
