from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Enhanced Image Captioning Workflow')

# Define node styles with gradient and colors
# Start and End nodes with oval shape
dot.attr('node', shape='ellipse', style='filled', fontname='Roboto', fontsize='20', fontcolor='white', fillcolor='lightblue:deepskyblue', color='transparent', width='3.0', height='1.5')
dot.node('A', 'Start\n(Import Libraries)')
dot.node('P', 'End')

# Middle nodes with rounded rectangle shape
dot.attr('node', shape='box', style='rounded,filled', fontname='Roboto', fontsize='16', fontcolor='white', color='transparent', width='3.5', height='1.8')

# Colors for middle nodes
dot.node('B', 'Load COCO Dataset', fillcolor='lightgreen:darkgreen')
dot.node('C', 'Build Vocabulary\n(build_coco_vocab)', fillcolor='lightsteelblue:royalblue')
dot.node('D', 'Tokenize Captions\n(tokenize_caption)', fillcolor='lightcoral:darkred')
dot.node('E', 'Collate Function\n(coco_collate_fn)', fillcolor='lightsteelblue:royalblue')
dot.node('F', 'Create DataLoader\n(get_dataloader)', fillcolor='lightpink:deeppink')

dot.node('G', 'Define Models\n(ImageCaptioningModel, SCSTModel, SATModel)', fillcolor='cyan:darkcyan')

dot.node('H', 'Train NIC Model\n(train_and_validate_model)', fillcolor='lightseagreen:seagreen')
dot.node('I', 'Evaluate NIC Model\n(evaluate_model_with_rouge_and_bleu)', fillcolor='yellow:goldenrod')

dot.node('J', 'Train SCST Model\n(train_and_validate_model)', fillcolor='lightsalmon:salmon')
dot.node('K', 'Evaluate SCST Model\n(evaluate_model_with_rouge_and_bleu)', fillcolor='lightgreen:yellowgreen')

dot.node('L', 'Train SAT Model\n(train_and_validate_model)', fillcolor='lightblue:skyblue')
dot.node('M', 'Evaluate SAT Model\n(evaluate_model_with_rouge_and_bleu)', fillcolor='lightgray:darkgray')

# Define edges with colors and styles
dot.attr('edge', color='deepskyblue', fontname='Roboto', fontsize='12', fontcolor='black')

dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')

dot.edge('F', 'G')

# Branches for model training and evaluation
dot.edge('G', 'H', label='Train NIC', color='limegreen', style='dashed')
dot.edge('H', 'I', label='Evaluate NIC', color='limegreen', style='dashed')

dot.edge('G', 'J', label='Train SCST', color='orange', style='dashed')
dot.edge('J', 'K', label='Evaluate SCST', color='orange', style='dashed')

dot.edge('G', 'L', label='Train SAT', color='purple', style='dashed')
dot.edge('L', 'M', label='Evaluate SAT', color='purple', style='dashed')

# Paths to results display and metrics saving
dot.edge('I', 'N', label='Results', color='black')
dot.edge('K', 'N', label='Results', color='black')
dot.edge('M', 'N', label='Results', color='black')

dot.edge('N', 'O', color='black')
dot.edge('O', 'P', color='black')

# Additional nodes for result display and saving metrics
dot.attr('node', shape='ellipse', style='filled', fontname='Roboto', fontsize='20', fontcolor='white', fillcolor='coral:darkred', color='transparent', width='3.0', height='1.5')
dot.node('N', 'Display Results\n(display_image_and_captions)')

dot.attr('node', shape='ellipse', style='filled', fontname='Roboto', fontsize='20', fontcolor='white', fillcolor='gold:orange', color='transparent', width='3.0', height='1.5')
dot.node('O', 'Save Metrics\n(save_metrics_to_json)')

# Render and save the flowchart
dot.render('modern_image_captioning_flowchart', format='png', cleanup=True)

print("Modern flowchart has been generated and saved as 'modern_image_captioning_flowchart.png'")
