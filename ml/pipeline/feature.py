class Feature:
    def __init__(self):
        is_featuretools = False

        # fat_content_dict = {'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}
        # raw_data['Item_Fat_Content'] = raw_data['Item_Fat_Content'].replace(fat_content_dict, regex=True)

        # raw_data["id"] = raw_data["Item_Identifier"] + raw_data["Outlet_Identifier"]
        # raw_data.drop(["Item_Identifier"], axis=1, inplace=True)
        # raw_data = raw_data.dropna(axis=0)
    def _construct_entity_set(self):
        """
        Construct and dump the EntitySet
        """
        raw_data = self._get_raw_data()

        es = ft.EntitySet(id="sales")
        es.entity_from_dataframe(entity_id="bigmart", dataframe=raw_data, index="id")

        es.normalize_entity(
            base_entity_id="bigmart",
            new_entity_id="outlet",
            index="Outlet_Identifier",
            additional_variables=[
                "Outlet_Establishment_Year",
                "Outlet_Size",
                "Outlet_Location_Type",
                "Outlet_Type",
            ],
        )

        # TODO: dump the es file for future featuretools
        print(es)
        return es

    @cached_property
    def _feature_matrix(self):
        """
        Use `featuretools` to generate the `feature_matrix` and
        format the feature matrix according to the model
        """
        es = self._construct_entity_set()
        feature_matrix, feature_defs = ft.dfs(
            entityset=es, target_entity="bigmart", max_depth=2, verbose=1, n_jobs=1
        )

        # TODO: move to the pipeline in the future.
        new_feature_matrix = self._format_feature_matrix(feature_matrix)
        return new_feature_matrix

def _format_feature_matrix(self, feature_matrix):
    # categorical_features = np.where(feature_matrix.dtypes == 'object')[0]
    # for i in categorical_features:
    #     feature_matrix.iloc[:,i] = feature_matrix.iloc[:,i].astype('str')
    feature_matrix.drop(["Outlet_Identifier"], axis=1, inplace=True)

    return feature_matrix