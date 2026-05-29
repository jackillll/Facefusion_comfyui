"""
Local face swapping main function.
"""
from typing import Optional, List, Tuple

from .utils import VisionFrame, Face, find_matching_faces
from .detection.detector import detect_faces
from .models import get_local_swapper, get_face_occluder, get_face_parser


def swap_faces_local(
    source_image: VisionFrame,
    target_image: VisionFrame,
    model_name: str = 'hyperswap_1c_256',
    pixel_boost: str = '512x512',
    face_mask_blur: float = 0.3,
    face_selector_mode: str = 'one',
    source_face_index: int = 0,
    target_face_index: int = 0,
    source_sort_order: str = 'large-small',
    target_sort_order: str = 'large-small',
    score_threshold: float = 0.3,
    face_occluder_model: Optional[str] = None,
    face_parser_model: Optional[str] = None,
    face_detector_model: str = 'scrfd',
    face_mask_types: Optional[List[str]] = None,
    face_mask_areas: Optional[List[str]] = None,
    face_mask_regions: Optional[List[str]] = None,
    face_mask_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    reference_image: Optional[VisionFrame] = None,
    reference_face_distance: float = 0.6,
    source_face: Optional[Face] = None,
    reference_face: Optional[Face] = None,
    target_with_embedding: Optional[bool] = None,
    target_use_cache: bool = True,
    prepared_source_embedding = None
) -> VisionFrame:
    """
    Swap faces locally using ONNX models.
    
    Args:
        source_image: Source face image
        target_image: Target image to swap faces in
        model_name: Face swapper model to use
        pixel_boost: Resolution for pixel boost (256x256, 512x512, 768x768, 1024x1024)
        face_mask_blur: Blur amount for mask blending (0.0-1.0, default 0.3)
        face_selector_mode: How to select faces ('one', 'many', 'reference')
        source_face_index: Which face to use from source image
        target_face_index: Which face to use from target image when mode='one'
        source_sort_order: How to sort source faces before selecting
        target_sort_order: How to sort target faces before selecting
        score_threshold: Minimum detection confidence
        face_occluder_model: Face occluder model to use (xseg_1, xseg_2, xseg_3) for masking occlusions
        face_parser_model: Face parser model to use (bisenet_resnet_18, bisenet_resnet_34) for region segmentation
        face_detector_model: Face detector model to use (scrfd, retinaface, yolo_face, yunet, many)
        face_mask_types: List of mask types to use ['box', 'occlusion', 'area', 'region']
        face_mask_areas: List of face areas for area mask ['upper-face', 'lower-face', 'mouth']
        face_mask_regions: List of face regions for region mask ['skin', 'nose', 'mouth', etc.]
        face_mask_padding: Padding for box mask (top, right, bottom, left)
        reference_image: Reference image used when face_selector_mode='reference'
        reference_face_distance: Embedding distance threshold for reference matching
    
    Returns:
        Image with swapped faces
    """
    # Default mask types if not specified
    if face_mask_types is None:
        face_mask_types = ['box']
    # print(f"[LocalSwap] Starting local face swap with model: {model_name}")
    
    if target_with_embedding is None:
        target_with_embedding = face_selector_mode == 'reference'

    if source_face is None:
        source_faces = detect_faces(source_image, score_threshold, source_sort_order, face_detector_model, with_embedding=True)
        if not source_faces:
            # print("[LocalSwap] No faces detected in source image")
            return target_image
        source_face = source_faces[min(source_face_index, len(source_faces) - 1)]

    target_faces = detect_faces(
        target_image,
        score_threshold,
        target_sort_order,
        face_detector_model,
        with_embedding=target_with_embedding,
        use_cache=target_use_cache
    )
    
    if not target_faces:
        # print("[LocalSwap] No faces detected in target image")
        return target_image
    
    # Get swapper
    swapper = get_local_swapper(model_name)
    
    # Get occluder and parser if specified
    occluder = None
    parser = None
    if face_occluder_model and face_occluder_model != 'none':
        occluder = get_face_occluder(face_occluder_model)
    if face_parser_model and face_parser_model != 'none':
        parser = get_face_parser(face_parser_model)
    
    # Select target faces.
    if face_selector_mode == 'many':
        selected_target_faces = target_faces
    elif face_selector_mode == 'reference':
        if reference_face is None and reference_image is not None:
            reference_faces = detect_faces(reference_image, score_threshold, target_sort_order, face_detector_model, with_embedding=True)
            if reference_faces:
                reference_face = reference_faces[0]
        if reference_face is None:
            return target_image
        selected_target_faces = find_matching_faces(reference_face, target_faces, reference_face_distance)
    else:
        selected_target_faces = [target_faces[min(target_face_index, len(target_faces) - 1)]]

    if not selected_target_faces:
        return target_image

    # Swap faces - pass occluder and parser instances
    result = target_image.copy()
    for target_face in selected_target_faces:
        result = swapper.swap_face(
            source_face, target_face, result, pixel_boost, face_mask_blur,
            occluder, parser, source_image,
            face_mask_types, face_mask_areas, face_mask_regions, face_mask_padding,
            prepared_source_embedding
        )
    
    # print("[LocalSwap] Face swap completed")
    return result

