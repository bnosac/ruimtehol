## CHANGES IN ruimtehol VERSION 0.2

- Allow to do transfer learning by passing an embedding matrix and keep on training based on that matrix 
- Allow to do semi-supervised learning easily with embed_tagspace
- Attributes attached to a model are now also restored when loading a model with starspace_load_model of type 'ruimtehol'
- Add range.textspace to get the range of embedding similarities

## CHANGES IN ruimtehol VERSION 0.1.2

- Changes to src/Makevars
    - Added -pthread in PKG_CPPFLAGS and removed usage of SHLIB_PTHREAD_FLAGS

## CHANGES IN ruimtehol VERSION 0.1.1

- Initial release based on STARSPACE-2017-2.
