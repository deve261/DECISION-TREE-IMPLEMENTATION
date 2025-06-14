import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

class DecisionTreeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Decision Tree Classifier")
        self.root.geometry("700x600")
        self.root.configure(bg="#f4f4f4")

        self.data = None
        self.target_col = tk.StringVar()
        self.test_size = tk.DoubleVar(value=0.3)

        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Decision Tree Classifier UI", bg="#f4f4f4", font=("Arial", 16)).pack(pady=10)

        self.file_label = tk.Label(self.root, text="No file loaded", bg="#f4f4f4")
        self.file_label.pack()

        tk.Button(self.root, text="Load CSV File", command=self.load_file, bg="#3498db", fg="white", width=20).pack(pady=5)

        self.dropdown = ttk.Combobox(self.root, textvariable=self.target_col, state="readonly")
        self.dropdown.pack(pady=5)

        tk.Label(self.root, text="Test Size (0.1 to 0.5):", bg="#f4f4f4").pack(pady=2)
        self.slider = tk.Scale(self.root, from_=0.1, to=0.5, resolution=0.05, orient="horizontal", variable=self.test_size)
        self.slider.pack()

        tk.Button(self.root, text="Train and Visualize", command=self.train_model, bg="#2ecc71", fg="white", width=20).pack(pady=10)

        tk.Button(self.root, text="Export Tree as Image", command=self.export_image, bg="#e67e22", fg="white", width=20).pack(pady=2)

        tk.Button(self.root, text="Predict New Record", command=self.predict_new_record, bg="#9b59b6", fg="white", width=20).pack(pady=5)

        self.accuracy_label = tk.Label(self.root, text="Accuracy: N/A", bg="#f4f4f4", font=("Arial", 12))
        self.accuracy_label.pack(pady=5)

        self.output_text = tk.Text(self.root, height=15, width=80)
        self.output_text.pack(pady=10)

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            try:
                self.data = pd.read_csv(filepath)
                self.file_label.config(text=f"Loaded: {filepath.split('/')[-1]}")
                self.dropdown['values'] = list(self.data.columns)
                self.target_col.set(self.data.columns[-1])  # Default to last column
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def train_model(self):
        if self.data is None or self.target_col.get() not in self.data.columns:
            messagebox.showerror("Error", "Please load a dataset and select a target column.")
            return

        try:
            X = self.data.drop(columns=[self.target_col.get()])
            y = self.data[self.target_col.get()]

            if X.select_dtypes(include=[object]).shape[1] > 0:
                X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size.get(), random_state=42)

            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Classification Report:\n{report}")
            self.accuracy_label.config(text=f"Accuracy: {acc:.2%}")

            self.last_tree = clf
            self.last_features = X.columns
            self.last_classes = list(map(str, np.unique(y)))
            self.visualize_tree()

        except Exception as e:
            messagebox.showerror("Error", f"Model training failed: {e}")

    def visualize_tree(self):
        plt.figure(figsize=(12, 6))
        plot_tree(self.last_tree, filled=True, feature_names=self.last_features, class_names=self.last_classes)
        plt.title("Decision Tree")
        plt.show()

    def export_image(self):
        if hasattr(self, 'last_tree'):
            filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if filename:
                plt.figure(figsize=(12, 6))
                plot_tree(self.last_tree, filled=True, feature_names=self.last_features, class_names=self.last_classes)
                plt.title("Decision Tree")
                plt.savefig(filename)
                plt.close()
                messagebox.showinfo("Saved", f"Tree image saved as {filename}")
        else:
            messagebox.showerror("Error", "No tree to export. Train a model first.")

    def predict_new_record(self):
        if not hasattr(self, 'last_tree'):
            messagebox.showerror("Error", "Train the model first.")
            return

        form = tk.Toplevel(self.root)
        form.title("Enter New Record")
        form.geometry("300x400")
        form.configure(bg="#f0f0f0")

        entries = {}
        input_frame = tk.Frame(form, bg="#f0f0f0")
        input_frame.pack(pady=10)

        for i, col in enumerate(self.last_features):
            label = tk.Label(input_frame, text=col, bg="#f0f0f0")
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry = tk.Entry(input_frame, width=25)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entries[col] = entry

        def make_prediction():
            try:
                input_data = []
                for col in self.last_features:
                    val = entries[col].get()
                    try:
                        input_data.append(float(val))
                    except:
                        input_data.append(val)

                df = pd.DataFrame([input_data], columns=self.last_features)
                prediction = self.last_tree.predict(df)[0]
                messagebox.showinfo("Prediction Result", f"Predicted Outcome: {prediction}")
                form.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")

        tk.Button(form, text="Predict", command=make_prediction, bg="#2ecc71", fg="white").pack(pady=20)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = DecisionTreeApp(root)
    root.mainloop()
