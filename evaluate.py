import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    accuracy_score, 
    top_k_accuracy_score,
    roc_auc_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
from pathlib import Path
from config import PathConfig
from utils.model_utils import L2Normalization
import tensorflow.keras.backend as K
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define distortion suffixes
DISTORTION_SUFFIXES = ['blurred', 'foggy', 'lowlight', 'noisy', 'rainy', 'resized', 'sunny']

def contrastive_loss(margin=1.0):
    """Contrastive loss function for siamese networks"""
    def loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

def contrastive_accuracy(margin=1.0):
    """Accuracy metric for contrastive loss"""
    def accuracy(y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
    return accuracy

def visualize_embeddings(embeddings, labels, output_path):
    """Create t-SNE visualization of embeddings"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(16, 12))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                   color=colors(i), label=label, alpha=0.7, s=50)
    
    plt.title('t-SNE Visualization of Face Embeddings', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_visual_report(y_true, y_pred, class_names, output_path, title):
    """Enhanced visual report with better formatting"""
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T
    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', cbar=False)
    plt.title(f"{title}\nClassification Report", pad=20)
    
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=False)
    plt.title("Confusion Matrix", pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.subplot(1, 3, 3)
    class_dist = pd.Series(y_true).value_counts().sort_index()
    class_dist.plot(kind='bar', color='skyblue')
    plt.title("Class Distribution", pad=20)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def save_individual_results(images, filenames, true_labels, pred_labels, knn, embeddings, output_dir):
    """Save individual results with confidence scores"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, (img, filename, true_label, pred_label) in enumerate(
        zip(images, filenames, true_labels, pred_labels)):
        
        person_dir = os.path.join(output_dir, true_label)
        os.makedirs(person_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 8))
        img_uint8 = (img * 255).astype('uint8') if img.dtype != np.uint8 else img
        plt.imshow(img_uint8)
        
        confidence = ""
        if knn and hasattr(knn, 'predict_proba'):
            try:
                proba = knn.predict_proba(embeddings[i:i+1])[0]
                pred = knn.predict(embeddings[i:i+1])[0]
                pred_idx = np.where(knn.classes_ == pred)[0][0]
                confidence = f"\nConfidence: {proba[pred_idx]:.2f}"
            except Exception as e:
                print(f"Error getting confidence score: {str(e)}")
        
        title = f"True: {true_label}\nPred: {pred_label}{confidence}"
        plt.title(title, fontsize=10)
        plt.axis('off')
        
        base_name = os.path.splitext(os.path.basename(filename))[0]
        img_path = os.path.join(person_dir, f"{base_name}_result.jpg")
        plt.savefig(img_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        results.append({
            'filename': filename,
            'true_label': true_label,
            'pred_label': pred_label,
            'correct': int(pred_label == true_label)
        })
    
    return results

def save_individual_gender_results(images, labels, preds, output_dir):
    """Save gender classification results with visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, (img, label, pred) in enumerate(zip(images, labels, preds)):
        plt.figure(figsize=(6, 6))
        plt.imshow((img * 255).astype('uint8'))
        
        true_label = 'female' if label == 0 else 'male'
        pred_label = 'female' if pred < 0.5 else 'male'
        confidence = pred[0] if pred_label == 'male' else 1 - pred[0]
        
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})")
        plt.axis('off')
        
        img_path = os.path.join(output_dir, f"result_{i:04d}.jpg")
        plt.savefig(img_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        results.append({
            'image_id': i,
            'true_label': true_label,
            'pred_label': pred_label,
            'confidence': float(confidence),
            'correct': int(pred_label == true_label)
        })
    
    return results

def evaluate_gender_classifier(model_path, val_gen, results_dir):
    """Evaluate gender classification model"""
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Gender model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    val_gen.reset()
    images, labels, preds = [], [], []
    for _ in range(len(val_gen)):
        batch = next(val_gen)
        images.extend(batch[0])
        labels.extend(batch[1])
        preds.extend(model.predict(batch[0], verbose=0))
    
    y_true = np.array(labels).squeeze()
    y_pred = (np.array(preds).squeeze() > 0.5).astype(int)
    roc_auc = roc_auc_score(y_true, np.array(preds).squeeze())
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    
    save_visual_report(
        y_true, y_pred, ['female', 'male'],
        os.path.join(results_dir, "report.jpg"),
        "Gender Classification"
    )
    
    results = save_individual_gender_results(
        images, labels, preds,
        os.path.join(results_dir, "individual"))
    
    accuracy = np.mean([r['correct'] for r in results])
    metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'num_samples': len(results)
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(results_dir, "metrics.csv"), index=False)
    
    print(f"\nGender Classification Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

class FaceRecognitionEvaluator:
    """Class to handle face recognition evaluation with KNN"""
    def __init__(self, embedding_network):
        self.embedding_network = embedding_network
        self.knn = None
        self.label_encoder = LabelEncoder()
        
    def process_images(self, directory):
        """Process all images in the directory and extract embeddings"""
        embeddings, labels, images, filenames = [], [], [], []
        
        for person_id in tqdm(sorted(os.listdir(directory)), desc="Processing identities"):
            person_path = os.path.join(directory, person_id)
            if not os.path.isdir(person_path):
                continue
                
            # Process clean images
            for file in sorted(os.listdir(person_path)):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                if file.startswith('.'):
                    continue
                if file.lower() == 'distortion':
                    continue
                    
                clean_img_path = os.path.join(person_path, file)
                img, embedding, filename = self._process_single_image(clean_img_path)
                if embedding is not None:
                    images.append(img)
                    embeddings.append(embedding)
                    labels.append(person_id)
                    filenames.append(filename)
            
            # Process distorted images
            distortions_dir = os.path.join(person_path, 'distortion')
            if os.path.exists(distortions_dir):
                for dist_file in sorted(os.listdir(distortions_dir)):
                    if not dist_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    dist_path = os.path.join(distortions_dir, dist_file)
                    img, embedding, filename = self._process_single_image(dist_path)
                    if embedding is not None:
                        images.append(img)
                        embeddings.append(embedding)
                        labels.append(person_id)
                        filenames.append(filename)
        
        if not embeddings:
            raise ValueError("No valid images found for evaluation")
            
        return np.array(images), np.vstack(embeddings), np.array(labels), filenames
    
    def _process_single_image(self, img_path):
        """Helper to process a single image file"""
        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img_np = tf.image.resize(img, (224, 224)).numpy()
            
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype('uint8')
            
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_np)
            embedding = self.embedding_network.predict(
                tf.expand_dims(img_preprocessed, axis=0), verbose=0)
            return img_np, embedding, img_path
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None, None, None
    
    def evaluate(self, images, embeddings, labels, filenames):
        """Run full evaluation pipeline"""
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Separate clean and distorted images
        clean_indices = [i for i, fname in enumerate(filenames) 
                        if 'distortion' not in fname.lower()]
        distorted_indices = [i for i in range(len(filenames)) 
                           if i not in clean_indices]
        
        if not clean_indices:
            raise ValueError("No clean images found for training")
        if not distorted_indices:
            raise ValueError("No distorted images found for testing")
        
        print(f"Using {len(clean_indices)} clean images for training")
        print(f"Using {len(distorted_indices)} distorted images for testing")
        
        # Train on clean images only
        X_train = embeddings[clean_indices]
        y_train = y_encoded[clean_indices]
        
        # Test on distorted images only
        X_test = embeddings[distorted_indices]
        y_test = y_encoded[distorted_indices]
        img_test = images[distorted_indices]
        filename_test = [filenames[i] for i in distorted_indices]
        
        self.knn = KNeighborsClassifier(
            n_neighbors=5,
            metric='cosine',
            weights='distance'
        )
        self.knn.fit(X_train, y_train)
        
        y_pred = self.knn.predict(X_test)
        y_proba = self.knn.predict_proba(X_test) if hasattr(self.knn, 'predict_proba') else None
        
        metrics = {
        'top1_accuracy': accuracy_score(y_test, y_pred),
        'f1_score_macro': f1_score(y_test, y_pred, average='macro'),
        'num_classes': len(self.label_encoder.classes_)
    }
        
        if y_proba is not None:
            present_classes = np.unique(y_test)
            valid_indices = [i for i, cls in enumerate(self.knn.classes_) 
                           if cls in present_classes]
            if len(valid_indices) > 0:
                y_proba_filtered = y_proba[:, valid_indices]
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba_filtered, multi_class='ovr')
            
            top_k = min(3, len(self.label_encoder.classes_))
            metrics[f'top_{top_k}_accuracy'] = top_k_accuracy_score(y_test, y_proba, k=top_k)
        
        y_test_names = self.label_encoder.inverse_transform(y_test)
        y_pred_names = self.label_encoder.inverse_transform(y_pred)
        
        return {
            'images': img_test,
            'filenames': filename_test,
            'true_labels': y_test_names,
            'pred_labels': y_pred_names,
            'embeddings': X_test,
            'metrics': metrics,
            'knn': self.knn
        }

def evaluate_face_recognition(model_path, data_dir, results_dir):
    """Main face recognition evaluation function"""
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Face recognition model not found at {model_path}")
    
    try:
        siamese_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'L2Normalization': L2Normalization,
                'contrastive_loss': contrastive_loss(),
                'contrastive_accuracy': contrastive_accuracy()
            },
            compile=False
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    try:
        embedding_network = siamese_model.layers[2]
    except Exception as e:
        print(f"Error extracting embedding network: {str(e)}")
        raise
    
    evaluator = FaceRecognitionEvaluator(embedding_network)
    
    print("\nProcessing validation images...")
    try:
        images, embeddings, labels, filenames = evaluator.process_images(data_dir)
        results = evaluator.evaluate(images, embeddings, labels, filenames)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
    
    save_visual_report(
        results['true_labels'], results['pred_labels'],
        evaluator.label_encoder.classes_,
        os.path.join(results_dir, "report.jpg"),
        "Face Recognition"
    )
    
    individual_results = save_individual_results(
        results['images'], results['filenames'],
        results['true_labels'], results['pred_labels'],
        results['knn'], results['embeddings'],
        os.path.join(results_dir, "individual"))
    
    pd.DataFrame(results['metrics'], index=[0]).to_csv(
        os.path.join(results_dir, "metrics.csv"), index=False)
    pd.DataFrame(individual_results).to_csv(
        os.path.join(results_dir, "predictions.csv"), index=False)
    
    visualize_embeddings(embeddings, labels,
                        os.path.join(results_dir, "embeddings.jpg"))
    
    print("\nFace Recognition Results:")
    for metric, value in results['metrics'].items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

if __name__ == "__main__":
    print("Starting evaluation pipeline...")
    
    try:
        from utils.data_utils import load_gender_classification_data
        _, val_gen = load_gender_classification_data()
        
        print("\n=== Evaluating Gender Classification ===")
        evaluate_gender_classifier(
            str(PathConfig.GENDER_MODEL),
            val_gen,
            str(PathConfig.GENDER_RESULTS)
        )
        
        print("\n=== Evaluating Face Recognition ===")
        evaluate_face_recognition(
            str(PathConfig.FACE_MODEL), 
            str(PathConfig.TASK_B_VAL),
            str(PathConfig.FACE_RESULTS)
        )
        
    except Exception as e:
        print(f"\nEvaluation failed with error: {str(e)}")
    
    print(f"\nEvaluation complete! Results saved to {PathConfig.RESULTS}")