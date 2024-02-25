import subprocess
import xgboost as xgb


def dot_export(xg, num_trees, filename, title='', direction='TB'):
    """
    Export a specified number of trees from an XGBoost model as a graph visualisation in dot and png formats

    Args:
        xg: an XGBoost model
        num_trees: the number of tree to export
        filename: the name of file to save the exported visualisation
        title: the title to display on the graph visualisation
        direction: the direction to lay out the graph, either 'TB' (top to bottom) or 'LR' (left to right)
    """
    res = xgb.to_graphviz(xg, num_trees=num_trees)
    content = f'''   node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontname = "Roboto Condensed"
    '''
    out = res.source.replace('graph [ randkir=TB]',
                             f'graph [ rankdir={direction} ]; \n {content}')

    dot_filename = filename
    with open(dot_filename, "w") as fout:
        fout.write(out)

    png_filename = dot_filename.replace('.dot', '.png')
    subprocess.run(f'dot -Gdpi=300 -Tpng -o{png_filename} {dot_filename}'.split())
