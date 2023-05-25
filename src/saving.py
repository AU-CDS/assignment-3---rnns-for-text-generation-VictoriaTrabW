#loading data and appending the comments section data to a list
#this is the original loading the entire dataset
all_comments = []
for filename in os.listdir(data_dir):
    if 'Comments' in filename:
        comments_df = pd.read_csv(data_dir + "/" + filename)
        all_comments.extend(list(comments_df["commentBody"].values))