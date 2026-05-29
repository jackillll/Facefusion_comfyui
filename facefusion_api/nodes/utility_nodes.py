"""
Utility Nodes for ComfyUI.
"""
from .base import *
from .image_nodes import SwapFaceImage

class PixelBoostNode:
	"""Node for setting pixel boost resolution (for local face swapping)."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'image': (IO.IMAGE,),
				'pixel_boost':
				(
					['256x256', '512x512', '768x768', '1024x1024'],
					{
						'default': '512x512'
					}
				)
			}
		}
	
	RETURN_TYPES = (IO.IMAGE, 'STRING')
	RETURN_NAMES = ('image', 'pixel_boost_setting')
	FUNCTION = 'process'
	CATEGORY = 'FaceFusion'
	
	def process(self, image: Tensor, pixel_boost: str) -> Tuple[Tensor, str]:
		"""Pass through image and pixel boost setting."""
		# This node serves as a configuration node for pixel boost settings
		# The actual pixel boost processing happens in the face swapping nodes
		# print(f"[PixelBoostNode] Setting: {pixel_boost}")
		return (image, pixel_boost)




class FaceSwapApplier:
	"""Node to apply face swap to specific detected faces."""
	
	@classmethod
	def INPUT_TYPES(s) -> InputTypes:
		return\
		{
			'required':
			{
				'source_images': (IO.IMAGE,),
				'target_face_data': ('FACE_DATA',),
				'face_swapper_model':
				(
					[
						'hyperswap_1a_256',
						'hyperswap_1b_256',
						'hyperswap_1c_256'
					],
					{
						'default': 'hyperswap_1c_256'
					}
				),
				'pixel_boost':
				(
					['256x256', '512x512', '768x768', '1024x1024'],
					{
						'default': '512x512'
					}
				),
			'face_occluder_model':
			(
				['none', 'xseg_1', 'xseg_2', 'xseg_3'],
				{
					'default': 'xseg_1'
				}
			),
			'face_parser_model':
			(
				['none', 'bisenet_resnet_18', 'bisenet_resnet_34'],
				{
					'default': 'bisenet_resnet_34'
				}
			),
				'face_mask_blur':
				(
					'FLOAT',
					{
						'default': 0.3,
						'min': 0.0,
						'max': 1.0,
						'step': 0.05
					}
				),
				'face_index':
				(
					'INT',
					{
						'default': 0,
						'min': 0,
						'max': 100,
						'step': 1
					}
				),
				'enable_nsfw_check':
				(
					'BOOLEAN',
					{
						'default': True
					}
				)
			}
		}
	
	RETURN_TYPES = (IO.IMAGE, 'FACE_DATA')
	RETURN_NAMES = ('swapped_image', 'face_data')
	FUNCTION = 'apply'
	CATEGORY = 'FaceFusion'
	
	def apply(
		self,
		source_images: Tensor,
		target_face_data: Dict,
		face_swapper_model: FaceSwapperModel,
		pixel_boost: str,
		face_occluder_model: str,
		face_parser_model: str,
		face_mask_blur: float,
		face_index: int,
		enable_nsfw_check: bool = True
	) -> Tuple[Tensor, Dict]:
		"""Apply face swap to specific detected face - smart batch handling."""
		try:
			# Get target image
			target_image = target_face_data.get('image')
			faces = target_face_data.get('faces', [])
			
			if not faces:
				print("No faces in face_data to swap")
				return (target_image, target_face_data)
			
			if face_index >= len(faces):
				print(f"Face index {face_index} out of range (only {len(faces)} faces detected)")
				face_index = 0
			
			# Handle multiple source images
			if source_images.dim() == 4 and source_images.shape[0] > 1:
				source_image = source_images[0:1]
			else:
				source_image = source_images

			# Use the detected face selection directly for local inference.
			if target_image.dim() == 4 and target_image.shape[0] == 1:
				source_cv2 = tensor_to_cv2(source_image)
				target_cv2 = tensor_to_cv2(target_image)

				if enable_nsfw_check and CONTENT_FILTER_AVAILABLE:
					if analyse_frame(source_cv2) or analyse_frame(target_cv2):
						print("[ContentFilter] NSFW content detected - returning blurred output")
						return (cv2_to_tensor(blur_frame(target_cv2)), target_face_data)

				source_faces = detect_faces(source_cv2)
				if not source_faces:
					print("No source faces detected")
					return (target_image, target_face_data)

				selected_face = faces[face_index]
				target_face = {
					'bbox': np.asarray(selected_face['bbox'], dtype=np.float32),
					'landmarks': np.asarray(selected_face['landmarks'], dtype=np.float32),
					'score': float(selected_face.get('score', 0.0)),
					'area': float(selected_face.get('area', 0.0)),
				}
				if selected_face.get('embedding') is not None:
					target_face['embedding'] = np.asarray(selected_face['embedding'], dtype=np.float32)
				if selected_face.get('embedding_norm') is not None:
					target_face['embedding_norm'] = np.asarray(selected_face['embedding_norm'], dtype=np.float32)

				occluder = None if face_occluder_model == 'none' else get_face_occluder(face_occluder_model)
				parser = None if face_parser_model == 'none' else get_face_parser(face_parser_model)
				swapper = get_local_swapper(face_swapper_model)
				swapped_cv2 = swapper.swap_face(
					source_faces[0],
					target_face,
					target_cv2,
					pixel_boost,
					face_mask_blur,
					occluder,
					parser,
					source_cv2,
					['box']
				)
				return (cv2_to_tensor(swapped_cv2), target_face_data)
			
			# Smart batch handling for target images
			if target_image.dim() == 4 and target_image.shape[0] > 1:
				# Process batch
				print(f"[FaceSwapApplier] Processing batch of {target_image.shape[0]} images")
				output_images = []
				for i in range(target_image.shape[0]):
					single_target = target_image[i:i+1]
					swapped = SwapFaceImage.swap_face(
						source_image, 
						single_target, 
						face_swapper_model, 
						pixel_boost, 
						face_mask_blur,
						face_occluder_model,
						face_parser_model,
						enable_nsfw_check=enable_nsfw_check
					)
					output_images.append(swapped)
				swapped_image = torch.cat(output_images, dim=0)
			else:
				# Single image
				swapped_image = SwapFaceImage.swap_face(
					source_image, 
					target_image, 
					face_swapper_model, 
					pixel_boost, 
					face_mask_blur,
					face_occluder_model,
					face_parser_model,
					enable_nsfw_check=enable_nsfw_check
				)
			
			print(f"Applied face swap to face {face_index}")
			
			return (swapped_image, target_face_data)
		except Exception as e:
			print(f"Error applying face swap: {e}")
			import traceback
			traceback.print_exc()
			return (target_face_data.get('image'), target_face_data)
