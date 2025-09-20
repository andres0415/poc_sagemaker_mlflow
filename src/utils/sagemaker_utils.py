"""
SageMaker integration utilities for MLflow models.
"""
import boto3
import mlflow
import mlflow.sagemaker as mfs
from typing import Dict, Any, Optional
import json
import os

class SageMakerMLflowIntegration:
    """
    Utility class for integrating MLflow models with SageMaker.
    """
    
    def __init__(self, region_name: str = None, role_arn: str = None):
        """
        Initialize SageMaker integration.
        
        Args:
            region_name: AWS region name
            role_arn: SageMaker execution role ARN
        """
        self.region_name = region_name or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.role_arn = role_arn or os.getenv('SAGEMAKER_ROLE')
        
        if not self.role_arn:
            raise ValueError("SageMaker execution role ARN must be provided")
        
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
    
    def deploy_model(self, model_uri: str, endpoint_name: str,
                    instance_type: str = 'ml.t2.medium',
                    instance_count: int = 1) -> Dict[str, Any]:
        """
        Deploy MLflow model to SageMaker endpoint.
        
        Args:
            model_uri: MLflow model URI (e.g., "models:/model_name/version")
            endpoint_name: Name for the SageMaker endpoint
            instance_type: SageMaker instance type
            instance_count: Number of instances
            
        Returns:
            Deployment information
        """
        try:
            deployment_info = mfs.deploy(
                model_uri=model_uri,
                region_name=self.region_name,
                mode="create",
                execution_role_arn=self.role_arn,
                instance_type=instance_type,
                instance_count=instance_count,
                endpoint_name=endpoint_name
            )
            
            return {
                'status': 'success',
                'endpoint_name': endpoint_name,
                'model_uri': model_uri,
                'deployment_info': deployment_info
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'endpoint_name': endpoint_name,
                'model_uri': model_uri
            }
    
    def create_model_package(self, model_uri: str, model_package_name: str,
                           description: str = None) -> Dict[str, Any]:
        """
        Create a SageMaker Model Package from MLflow model.
        
        Args:
            model_uri: MLflow model URI
            model_package_name: Name for the model package
            description: Model package description
            
        Returns:
            Model package creation result
        """
        # This is a simplified example - actual implementation would require
        # more detailed configuration for model packages
        try:
            # Get model information from MLflow
            model_info = mlflow.models.get_model_info(model_uri)
            
            # Create model package group if it doesn't exist
            try:
                self.sagemaker_client.create_model_package_group(
                    ModelPackageGroupName=model_package_name,
                    ModelPackageGroupDescription=description or f"Model package for {model_uri}"
                )
            except self.sagemaker_client.exceptions.ValidationException:
                # Model package group already exists
                pass
            
            return {
                'status': 'success',
                'model_package_name': model_package_name,
                'model_uri': model_uri,
                'model_info': model_info.to_dict()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'model_package_name': model_package_name,
                'model_uri': model_uri
            }
    
    def list_endpoints(self) -> list:
        """
        List all SageMaker endpoints in the region.
        
        Returns:
            List of endpoint information
        """
        try:
            response = self.sagemaker_client.list_endpoints()
            return response.get('Endpoints', [])
        except Exception as e:
            print(f"Error listing endpoints: {e}")
            return []
    
    def delete_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Delete a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to delete
            
        Returns:
            Deletion result
        """
        try:
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # Delete endpoint configuration
            endpoint_config_name = f"{endpoint_name}-config"
            self.sagemaker_client.delete_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            
            return {
                'status': 'success',
                'endpoint_name': endpoint_name,
                'message': 'Endpoint deleted successfully'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'endpoint_name': endpoint_name
            }

def setup_sagemaker_mlflow_tracking(tracking_server_arn: str) -> None:
    """
    Configure MLflow to use SageMaker tracking server.
    
    Args:
        tracking_server_arn: ARN of the SageMaker MLflow tracking server
    """
    # Extract tracking URI from ARN
    # ARN format: arn:aws:sagemaker:region:account:mlflow-tracking-server/server-name
    try:
        region = tracking_server_arn.split(':')[3]
        server_name = tracking_server_arn.split('/')[-1]
        
        # Construct tracking URI (this would need to be the actual endpoint URL)
        tracking_uri = f"https://{server_name}.mlflow.{region}.amazonaws.com"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"MLflow tracking configured for SageMaker server: {tracking_uri}")
        
    except Exception as e:
        print(f"Error configuring SageMaker MLflow tracking: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMaker MLflow integration utilities")
    parser.add_argument("--action", choices=['deploy', 'list-endpoints', 'delete-endpoint'],
                       help="Action to perform")
    parser.add_argument("--model-uri", help="MLflow model URI")
    parser.add_argument("--endpoint-name", help="SageMaker endpoint name")
    parser.add_argument("--instance-type", default="ml.t2.medium", help="Instance type")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        print("SageMaker MLflow integration utilities")
        print("Use --help for available options")
        exit(0)
    
    integration = SageMakerMLflowIntegration()
    
    if args.action == "deploy" and args.model_uri and args.endpoint_name:
        result = integration.deploy_model(
            args.model_uri, 
            args.endpoint_name, 
            args.instance_type
        )
        print(json.dumps(result, indent=2))
        
    elif args.action == "list-endpoints":
        endpoints = integration.list_endpoints()
        print(json.dumps(endpoints, indent=2, default=str))
        
    elif args.action == "delete-endpoint" and args.endpoint_name:
        result = integration.delete_endpoint(args.endpoint_name)
        print(json.dumps(result, indent=2))