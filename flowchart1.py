from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Image Captioning Workflow')

# Define nodes with descriptions
dot.node('A', 'Start\n(Import Libraries)')
dot.node('B', 'Load COCO Dataset')
dot.node('C', 'Build Vocabulary\n(build_coco_vocab)')
dot.node('D', 'Tokenize Captions\n(tokenize_caption)')
dot.node('E', 'Collate Function\n(coco_collate_fn)')
dot.node('F', 'Create DataLoader\n(get_dataloader)')
dot.node('G', 'Define Models\n(ImageCaptioningModel, SCSTModel, SATModel)')
dot.node('H', 'Train Model\n(train_and_validate_model)')
dot.node('I', 'Evaluate Model\n(evaluate_model_with_rouge_and_bleu)')
dot.node('J', 'Display Results\n(display_image_and_captions)')
dot.node('K', 'Save Metrics\n(save_metrics_to_json)')
dot.node('L', 'End')

# Add edges between nodes
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')
dot.edge('J', 'K')
dot.edge('K', 'L')

# Render and save the flowchart
dot.render('image_captioning_flowchart', format='png', cleanup=True)

print("Flowchart has been generated and saved as 'image_captioning_flowchart.png'")


from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Image Captioning Workflow')

# Define nodes with descriptions
dot.node('A', 'Start\n(Import Libraries)')
dot.node('B', 'Load COCO Dataset\n(Initialize COCO, Build Vocabulary)')
dot.node('C', 'Preprocess Data\n(Transform Images, Tokenize Captions)')
dot.node('D', 'Create DataLoader\n(Load Dataset, Create Subset)')
dot.node('E', 'Define Models\n(NIC, SCST)')
dot.node('F', 'Train Models\n(Train NIC, Train SCST)')
dot.node('G', 'Evaluate Models\n(ROUGE, BLEU)')
dot.node('H', 'Test Image Captioning\n(Load Image, Predict Caption)')
dot.node('I', 'Display Results\n(GUI Elements)')
dot.node('J', 'End')

# Add edges between nodes
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')

# Render and save the flowchart
dot.render('image_captioning_workflow2', format='png', cleanup=True)

print("Flowchart has been generated and saved as 'image_captioning_workflow.png'")

